"""``operon keys``: model API keys in the OS credential store, e.g. Windows Credential Manager (KAN-186).

``set`` asks for the value with a hidden prompt; no command prints a value.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from operon.core import keystore
from operon.core.paths import real_environment

_NO_STORE = "no credential store available here; keep using .env or environment variables"


def _no_extra(args: argparse.Namespace) -> bool:
    if getattr(args, "extra", None):
        print("never pass a key value on the command line; the value is asked for, hidden", file=sys.stderr)
        return False
    return True


def _is_placeholder(value: str) -> bool:
    """The `your-...-here` text .env.example ships with."""
    low = value.strip().lower()
    return low.startswith("your-") and low.endswith("-here")


def _check_args(args: argparse.Namespace) -> bool:
    """Refuse extra words and unknown names without echoing either: a key may have been pasted there."""
    if not _no_extra(args):
        return False
    names = keystore.known_names()
    if args.name in names:
        return True
    print(f"unknown key name. Known names: {', '.join(names)}", file=sys.stderr)
    return False


def _store_ready() -> bool:
    if not keystore.enabled():
        print(f"{keystore.disabled_reason()}; keep using .env or environment variables", file=sys.stderr)
        return False
    if not keystore.available():
        print(_NO_STORE, file=sys.stderr)
        return False
    return True


def _set(args: argparse.Namespace) -> int:
    if not _check_args(args):
        return 2
    if not _store_ready():
        return 1
    value = getpass.getpass(f"Value for {args.name} (hidden): ")
    if getpass.getpass("Again to confirm: ") != value:
        print("The two entries differ; nothing was saved.", file=sys.stderr)
        return 1
    if not value.strip():
        print("Empty value; nothing was saved.", file=sys.stderr)
        return 1
    try:
        keystore.set(args.name, value)
    except keystore.KeystoreError as exc:
        print(f"Not saved: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {args.name} in the credential store.")
    if (real_environment().get(args.name) or "").strip():
        print(f"Note: the environment variable {args.name} is set and still wins until you remove it.")
    return 0


def _file_values(path: str | Path | None) -> dict[str, str | None]:
    from dotenv import dotenv_values

    if not path or not Path(path).is_file():
        return {}
    return dict(dotenv_values(path))


def _list(args: argparse.Namespace) -> int:
    if not _no_extra(args):
        return 2
    from dotenv import find_dotenv

    from operon.core.paths import loaded_key_state, user_env_file

    sources = loaded_key_state()[1]
    real = real_environment()  # the loader's first snapshot, or the environment as it is now
    dotenv = _file_values(find_dotenv(usecwd=True))
    user_file = _file_values(user_env_file())
    store = keystore.enabled() and keystore.available()
    for name in keystore.known_names():
        if (real.get(name) or "").strip():
            source = "environment"
        elif sources.get(name) == keystore.store_label() or (store and keystore.get(name)):
            source = keystore.store_label()
        elif (dotenv.get(name) or "").strip():
            source = ".env"
        elif (user_file.get(name) or "").strip():
            source = "operon.env"
        else:
            source = "not set"
        print(f"{name}: {source}")
    return 0


def _remove(args: argparse.Namespace) -> int:
    if not _check_args(args):
        return 2
    if not _store_ready():
        return 1
    try:
        removed = keystore.delete(args.name)
    except keystore.KeystoreError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not removed:
        print(f"No stored key named {args.name}.", file=sys.stderr)
        return 1
    print(f"Removed {args.name} from the credential store.")
    return 0


def _import(args: argparse.Namespace) -> int:
    if not _no_extra(args):
        return 2
    from dotenv import find_dotenv

    path = find_dotenv(usecwd=True)
    if not path:
        print("No .env in this folder or above; nothing to import.", file=sys.stderr)
        return 1
    if not _store_ready():
        return 1
    values = _file_values(path)
    copied: list[str] = []
    skipped: list[str] = []
    for name in keystore.known_names():
        value = (values.get(name) or "").strip()
        if not value:
            continue
        if _is_placeholder(value):
            skipped.append(name)
            continue
        try:
            keystore.set(name, value)
        except keystore.KeystoreError as exc:
            print(f"Not saved: {exc}", file=sys.stderr)
            return 1
        copied.append(name)
    if skipped:
        print(f"Skipped placeholder values (not real keys): {', '.join(skipped)}")
    if not copied:
        print(f"No model keys found in {path}.")
        return 0
    print(f"Copied to the credential store: {', '.join(copied)}")
    print(f"You can now delete these lines from {path} (Operon does not edit it):")
    for name in copied:
        print(f"  {name}=...")
    return 0


def add_parser(sub) -> None:
    parser = sub.add_parser("keys", help="store model API keys in the OS credential store instead of .env")
    actions = parser.add_subparsers(dest="keys_command", required=True)
    set_parser = actions.add_parser("set", help="store a key (the value is asked for, hidden)")
    set_parser.add_argument("name")
    set_parser.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    set_parser.set_defaults(func=_set)
    list_parser = actions.add_parser("list", help="show where each key comes from (never values)")
    list_parser.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    list_parser.set_defaults(func=_list)
    remove = actions.add_parser("rm", help="delete a stored key")
    remove.add_argument("name")
    remove.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    remove.set_defaults(func=_remove)
    import_parser = actions.add_parser("import", help="copy the keys in this folder's .env into the credential store")
    import_parser.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    import_parser.set_defaults(func=_import)
