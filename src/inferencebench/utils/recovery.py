import base64
import json
import lzma
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

MESSAGE_TYPES: dict[str, Any] = {
    "system": ChatMessageSystem,
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
    "tool": ChatMessageTool,
}


@solver
def restore_recovery_bundle(
    *,
    workspace_root: str,
    home_root: str,
    home_owner: str | None,
    bundle_path: str | None = None,
    bundle_url: str | None = None,
    bundle_b64: str | None = None,
    bundle_fernet_key: str | None = None,
    restore_workspace: bool = True,
    restore_home: bool = True,
    restore_store: bool = True,
    restore_messages: bool = True,
    continuation_prompt: str | None = None,
) -> Solver:
    """Restore a prior sample's files, store, and conversation before its agent starts."""
    sources = [source is not None for source in (bundle_path, bundle_url, bundle_b64)]
    if sum(sources) > 1:
        raise ValueError("Specify only one recovery bundle source")

    async def execute(state: TaskState, generate: Generate) -> TaskState:
        """Apply the configured recovery bundle to this sample state and sandbox."""
        if not any(sources):
            return state
        bundle = _load_bundle(
            bundle_path, bundle_url, bundle_b64, bundle_fernet_key=bundle_fernet_key
        )
        if restore_workspace:
            await _restore_files(bundle.get("workspace", {}), workspace_root)
        if restore_home:
            home_directories = await _restore_files(bundle.get("home", {}), home_root)
            if home_owner is not None:
                await _restore_file_owners(home_root, home_directories, home_owner)
        if restore_store:
            _restore_store(state, bundle.get("store", {}))
        if restore_messages:
            _restore_messages(state, bundle.get("messages", []), continuation_prompt)
        return state

    return execute


def _load_bundle(
    bundle_path: str | None,
    bundle_url: str | None,
    bundle_b64: str | None,
    bundle_fernet_key: str | None = None,
) -> dict[str, Any]:
    """Load a recovery bundle from a directory, zip archive, URL, or base64 zip."""
    if bundle_path is not None:
        path = Path(bundle_path)
        if path.is_dir():
            return _load_directory_bundle(path)
        return _load_zip_bundle(_decrypt_bundle(path.read_bytes(), bundle_fernet_key))
    if bundle_url is not None:
        with urllib.request.urlopen(bundle_url) as response:
            return _load_zip_bundle(_decrypt_bundle(response.read(), bundle_fernet_key))
    if bundle_b64 is not None:
        return _load_zip_bundle(
            _decrypt_bundle(
                base64.b64decode(bundle_b64, validate=True), bundle_fernet_key
            )
        )
    raise ValueError("A recovery bundle source is required")


def _decrypt_bundle(data: bytes, key: str | None) -> bytes:
    """Decrypt a Fernet-wrapped recovery bundle when the run supplies its key."""
    if key is None:
        return data
    try:
        from cryptography.fernet import Fernet, InvalidToken

        return Fernet(key.encode()).decrypt(data)
    except ImportError as error:
        raise RuntimeError(
            "An encrypted recovery bundle requires the 'cryptography' package."
        ) from error
    except (ValueError, InvalidToken) as error:
        raise ValueError("Recovery bundle decryption failed.") from error


def _load_directory_bundle(path: Path) -> dict[str, Any]:
    """Read an artifact-directory recovery bundle."""
    return {
        "workspace": _load_file_tree(path / "workspace"),
        "home": _load_file_tree(path / "home"),
        "store": _load_json_file(path / "grading_store.json", {}),
        "messages": _load_json_file(path / "messages.json", []),
        "manifest": _load_json_file(path / "manifest.json", {}),
    }


def _load_file_tree(path: Path) -> dict[str, str]:
    """Read all UTF-8 files below one bundle tree."""
    files: dict[str, str] = {}
    if path.exists():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                files[str(file_path.relative_to(path))] = file_path.read_text()
    return files


def _load_zip_bundle(data: bytes) -> dict[str, Any]:
    """Read a zip recovery bundle with file trees plus JSON sidecars."""
    if data.startswith(b"\xfd7zXZ\x00"):
        data = lzma.decompress(data)
    with zipfile.ZipFile(BytesIO(data)) as archive:
        names = set(archive.namelist())
        return {
            "workspace": _read_zip_tree(archive, names, "workspace"),
            "home": _read_zip_tree(archive, names, "home"),
            "store": _read_zip_json(archive, names, "grading_store.json", {}),
            "messages": _read_zip_json(archive, names, "messages.json", []),
            "manifest": _read_zip_json(archive, names, "manifest.json", {}),
        }


def _read_zip_tree(
    archive: zipfile.ZipFile, names: set[str], root: str
) -> dict[str, str]:
    """Read an allowlisted relative tree from a recovery zip."""
    files: dict[str, str] = {}
    prefix = f"{root}/"
    for name in sorted(names):
        if name.startswith(prefix) and not name.endswith("/"):
            relative = name.removeprefix(prefix)
            _validate_relative_path(relative)
            files[relative] = archive.read(name).decode()
    return files


def _load_json_file(path: Path, default: Any) -> Any:
    """Read JSON when present, otherwise return the supplied default."""
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _read_zip_json(
    archive: zipfile.ZipFile,
    names: set[str],
    name: str,
    default: Any,
) -> Any:
    """Read one JSON sidecar from a zip archive when present."""
    if name not in names:
        return default
    return json.loads(archive.read(name).decode())


async def _restore_files(files: dict[str, str], root: str) -> set[str]:
    """Write a bundled file tree into its configured sandbox root."""
    destination_root = PurePosixPath(root)
    top_level_directories: set[str] = set()
    for relative, content in files.items():
        _validate_relative_path(relative)
        destination = destination_root / relative
        await sandbox("default").write_file(destination.as_posix(), content)
        top_level_directories.add(PurePosixPath(relative).parts[0])
    return top_level_directories


async def _restore_file_owners(
    root: str, top_level_directories: set[str], owner: str
) -> None:
    """Give the normal agent account write access to restored home files."""
    destination_root = PurePosixPath(root)
    for directory in sorted(top_level_directories):
        result = await sandbox("default").exec(
            [
                "chown",
                "-R",
                owner,
                (destination_root / directory).as_posix(),
            ],
            user="root",
        )
        if not result.success:
            raise RuntimeError(
                f"Unable to set recovered file ownership for {directory}: "
                f"{result.stderr.strip()}"
            )


def _restore_store(state: TaskState, store_data: dict[str, Any]) -> None:
    """Copy serialized Inspect store keys into the current sample store."""
    for key, value in store_data.items():
        state.store.set(key, value)


def _restore_messages(
    state: TaskState,
    raw_messages: list[dict[str, Any]],
    continuation_prompt: str | None,
) -> None:
    """Replace the starting conversation with the recovered transcript and prompt."""
    if not raw_messages:
        return
    state.messages = _deserialize_messages(raw_messages)
    if continuation_prompt:
        state.messages.append(ChatMessageUser(content=continuation_prompt))


def _deserialize_messages(raw_messages: list[dict[str, Any]]) -> list[ChatMessage]:
    """Convert serialized Inspect chat messages back into typed message objects."""
    tool_result_ids = {
        message.get("tool_call_id")
        for message in raw_messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    restored: list[ChatMessage] = []
    for raw in raw_messages:
        message = dict(raw)
        role = message.get("role")
        if role == "assistant":
            message["tool_calls"] = [
                call
                for call in message.get("tool_calls") or []
                if call.get("id") in tool_result_ids
            ]
        message_type = MESSAGE_TYPES.get(str(role))
        if message_type is None:
            raise ValueError(
                f"Unsupported chat message role in recovery bundle: {role}"
            )
        restored.append(message_type.model_validate(message))
    return restored


def _validate_relative_path(path: str) -> None:
    """Reject absolute or parent-traversal paths inside recovery bundles."""
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"Invalid recovery file path: {path!r}")
