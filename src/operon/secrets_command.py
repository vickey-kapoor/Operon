"""``operon secrets``: the vault the model types from without seeing (KAN-103).

``set`` asks for the value with a hidden prompt: it is never a command-line argument
(that would land in shell history and the process list) and never printed.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from operon.core import vault


def _set(args: argparse.Namespace) -> int:
    if not vault.valid_name(args.name):
        print(f"Invalid name {args.name!r}: use 1-64 letters, digits, '_' or '-'.", file=sys.stderr)
        return 2
    value = getpass.getpass(f"Value for {args.name} (hidden): ")
    if getpass.getpass("Again to confirm: ") != value:
        print("The two entries differ; nothing was saved.", file=sys.stderr)
        return 1
    try:
        vault.set_secret(args.name, value)
    except vault.SecretError as exc:
        print(f"Not saved: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {args.name}. Use it as {{{{secret:{args.name}}}}} in a type action.")
    if not vault.secrets_enabled():
        print("Secrets are off: set OPERON_SECRETS=on for runs to use them.")
    return 0


def _list(_args: argparse.Namespace) -> int:
    for name in vault.list_secrets():
        print(name)
    return 0


def _remove(args: argparse.Namespace) -> int:
    try:
        removed = vault.delete_secret(args.name)
    except vault.SecretError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not removed:
        print(f"No secret named {args.name}.", file=sys.stderr)
        return 1
    print(f"Removed {args.name}.")
    return 0


def add_parser(sub) -> None:
    parser = sub.add_parser("secrets", help="store secrets the model can type without seeing them")
    actions = parser.add_subparsers(dest="secrets_command", required=True)
    set_parser = actions.add_parser("set", help="store a secret (the value is asked for, hidden)")
    set_parser.add_argument("name")
    set_parser.set_defaults(func=_set)
    actions.add_parser("list", help="list secret names (never values)").set_defaults(func=_list)
    remove = actions.add_parser("rm", help="delete a secret")
    remove.add_argument("name")
    remove.set_defaults(func=_remove)
