"""``operon keys``: model API keys in the OS credential store, e.g. Windows Credential Manager (KAN-186).

``set`` asks for the value with a hidden prompt; no command prints a value.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from operon.core import keystore

_NO_STORE = "no credential store available here; keep using .env or environment variables"


def _check_name(name: str) -> bool:
    names = keystore.known_names()
    if name in names:
        return True
    print(f"{name!r} is not a model key Operon reads. Known names: {', '.join(names)}", file=sys.stderr)
    return False


def _set(args: argparse.Namespace) -> int:
    if not _check_name(args.name):
        return 2
    if not keystore.available():
        print(_NO_STORE, file=sys.stderr)
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
    return 0


def _file_values(path: str | Path | None) -> dict[str, str | None]:
    from dotenv import dotenv_values

    if not path or not Path(path).is_file():
        return {}
    return dict(dotenv_values(path))


def _list(_args: argparse.Namespace) -> int:
    from dotenv import find_dotenv

    from operon.core.paths import loaded_key_state, user_env_file

    real, sources = loaded_key_state()
    if not real and not sources:
        real = dict(os.environ)  # the loader has not run in this process: the environment is what it is now
    dotenv = _file_values(find_dotenv(usecwd=True))
    user_file = _file_values(user_env_file())
    store = keystore.available()
    for name in keystore.known_names():
        if (real.get(name) or "").strip():
            source = "environment"
        elif sources.get(name) == "Credential Manager" or (store and keystore.get(name)):
            source = "Credential Manager"
        elif (dotenv.get(name) or "").strip():
            source = ".env"
        elif (user_file.get(name) or "").strip():
            source = "operon.env"
        else:
            source = "not set"
        print(f"{name}: {source}")
    return 0


def _remove(args: argparse.Namespace) -> int:
    if not _check_name(args.name):
        return 2
    try:
        removed = keystore.delete(args.name) if keystore.available() else False
    except keystore.KeystoreError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not removed:
        print(f"No stored key named {args.name}.", file=sys.stderr)
        return 1
    print(f"Removed {args.name} from the credential store.")
    return 0


def _import(_args: argparse.Namespace) -> int:
    from dotenv import find_dotenv

    path = find_dotenv(usecwd=True)
    if not path:
        print("No .env in this folder or above; nothing to import.", file=sys.stderr)
        return 1
    if not keystore.available():
        print(_NO_STORE, file=sys.stderr)
        return 1
    values = _file_values(path)
    copied: list[str] = []
    for name in keystore.known_names():
        value = (values.get(name) or "").strip()
        if not value:
            continue
        try:
            keystore.set(name, value)
        except keystore.KeystoreError as exc:
            print(f"Not saved: {exc}", file=sys.stderr)
            return 1
        copied.append(name)
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
    set_parser.set_defaults(func=_set)
    actions.add_parser("list", help="show where each key comes from (never values)").set_defaults(func=_list)
    remove = actions.add_parser("rm", help="delete a stored key")
    remove.add_argument("name")
    remove.set_defaults(func=_remove)
    actions.add_parser("import", help="copy the keys in this folder's .env into the credential store").set_defaults(func=_import)
