# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the project adapter contract with the standard library only.

The repository has no JSON Schema dependency, and a vendor runner must not acquire
one just to read its own contract.  This module implements the small keyword set the
contract schema actually uses and rejects any other keyword, so a schema that grows
a constraint this reader does not enforce fails loudly instead of passing silently.
"""

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ".flagcicd/project.schema.json"

_TYPES = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "boolean": lambda value: type(value) is bool,
    "integer": lambda value: type(value) is int,
}
_KEYWORDS = {
    "$id",
    "$schema",
    "$defs",
    "title",
    "description",
    "comment",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "pattern",
    "minLength",
    "minItems",
    "minProperties",
    "uniqueItems",
    "$ref",
}


def _resolve(reference, document):
    if not reference.startswith("#/"):
        raise ValueError(f"unsupported schema reference: {reference}")
    target = document
    for name in reference[2:].split("/"):
        if not isinstance(target, dict) or name not in target:
            raise ValueError(f"unresolved schema reference: {reference}")
        target = target[name]
    return target


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _check(value, schema, document, path):
    unknown = set(schema) - _KEYWORDS
    if unknown:
        raise ValueError(f"unsupported schema keyword at {path}: {sorted(unknown)}")
    if "$ref" in schema:
        _check(value, _resolve(schema["$ref"], document), document, path)
        return
    expected = schema.get("type")
    if expected is not None:
        if expected not in _TYPES:
            raise ValueError(f"unsupported schema type at {path}: {expected}")
        if not _TYPES[expected](value):
            raise ValueError(f"{path} must be {expected}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError(f"{path} must not be shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValueError(f"{path} does not match {schema['pattern']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path} must not hold fewer than {schema['minItems']}")
        if schema.get("uniqueItems") and len(
            {_canonical(item) for item in value}
        ) != len(value):
            raise ValueError(f"{path} must not repeat an item")
        if "items" in schema:
            for index, item in enumerate(value):
                _check(item, schema["items"], document, f"{path}[{index}]")
    if isinstance(value, dict):
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            raise ValueError(
                f"{path} must not hold fewer than {schema['minProperties']} entries"
            )
        for name in schema.get("required", ()):
            if name not in value:
                raise ValueError(f"{path} is missing {name}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in properties:
                _check(item, properties[name], document, f"{path}.{name}")
            elif additional is False:
                raise ValueError(f"{path} does not accept {name}")
            elif isinstance(additional, dict):
                _check(item, additional, document, f"{path}.{name}")


def validate_project(document, root=ROOT):
    """Reject a contract the core cannot interpret before anything reads its keys."""
    schema = json.loads((Path(root) / SCHEMA).read_text())
    _check(document, schema, schema, "project")
    return document
