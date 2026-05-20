# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import sys
import subprocess

import tempfile
import os
import time
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

SERVER_DIR = Path(__file__).resolve().parent
SAGE_ROOT = SERVER_DIR.parent


def _clean_env_value(value: str) -> str:
    value = value.strip().strip('"').strip("'")
    value = value.replace("[1m", "").rstrip("]")
    return value


def _load_dotenv(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value)
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def _load_key_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


env_dict = _load_dotenv(SAGE_ROOT / ".env")
key_dict = _load_key_json(SERVER_DIR / "key.json")


def _get_config(name: str, default: str = "") -> str:
    return os.getenv(name) or env_dict.get(name) or key_dict.get(name, default)


def _get_model_dict() -> dict:
    configured = key_dict.get("MODEL_DICT") or {}
    deepseek_model = _get_config("ANTHROPIC_MODEL", configured.get("claude") or configured.get("llm") or "deepseek-chat")
    qwen_model = _get_config("QWEN_MODEL", configured.get("qwen") or configured.get("vlm") or "qwen-vl-max-latest")
    model_dict = {
        "claude": deepseek_model,
        "llm": deepseek_model,
        "qwen": qwen_model,
        "vlm": qwen_model,
        "openai": _get_config("OPENAI_MODEL", configured.get("openai") or qwen_model),
        "glmv": _get_config("GLMV_MODEL", configured.get("glmv", qwen_model)),
    }
    model_dict.update({k: v for k, v in configured.items() if v})
    if _get_config("ANTHROPIC_MODEL"):
        model_dict["claude"] = _get_config("ANTHROPIC_MODEL")
        model_dict["llm"] = _get_config("ANTHROPIC_MODEL")
    if _get_config("QWEN_MODEL"):
        model_dict["qwen"] = _get_config("QWEN_MODEL")
        model_dict["vlm"] = _get_config("QWEN_MODEL")
    return model_dict

ANTHROPIC_API_KEY = _get_config("ANTHROPIC_API_KEY") or _get_config("ANTHROPIC_AUTH_TOKEN") or _get_config("DEEPSEEK_API_KEY")
ANTHROPIC_BASE_URL = _get_config("ANTHROPIC_BASE_URL") or _get_config("DEEPSEEK_ANTHROPIC_BASE_URL")
if ANTHROPIC_API_KEY:
    os.environ.setdefault("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
if ANTHROPIC_BASE_URL:
    os.environ.setdefault("ANTHROPIC_BASE_URL", ANTHROPIC_BASE_URL)

API_TOKEN = _get_config("API_TOKEN") or _get_config("QWEN_API_KEY") or _get_config("DASHSCOPE_API_KEY")
API_URL_QWEN = _get_config("API_URL_QWEN") or _get_config("QWEN_BASE_URL")
API_URL_OPENAI = _get_config("API_URL_OPENAI") or API_URL_QWEN

API_URL_DICT = {
    "qwen": API_URL_QWEN,
    "openai": API_URL_OPENAI,
    "glmv": API_URL_QWEN,
}

MODEL_DICT = _get_model_dict()

SERVER_URL = _get_config("TRELLIS_SERVER_URL", "http://localhost:8080")
FLUX_SERVER_URL = _get_config("FLUX_SERVER_URL")

print("TRELLIS SERVER_URL: ", SERVER_URL, file=sys.stderr)



def slurm_job_id_to_port(job_id, port_start=8080, port_end=40000):
    """
    Hash-based mapping function to convert SLURM job ID to a port number.
    
    Args:
        job_id (str or int): SLURM job ID
        port_start (int): Starting port number (default: 8080)
        port_end (int): Ending port number (default: 40000)
    
    Returns:
        int: Mapped port number within the specified range
    """
    # Convert job_id to string if it's an integer
    job_id_str = str(job_id)
    
    # Create a hash of the job ID
    hash_obj = hashlib.md5(job_id_str.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Map to port range
    port_range = port_end - port_start + 1
    mapped_port = port_start + (hash_int % port_range)
    
    return mapped_port


import requests
from loguru import logger
from requests.auth import HTTPBasicAuth

TOKEN_VALIDITY_SECONDS = 4 * 60 * 60 - 60  # 4 hours minus an error threshold of 1 min

last_client_refresh = {
    "oai": 0.0,
    "anthropic": 0.0,
}


def is_client_valid(api_service: str) -> bool:
    """
    Checks whether the API client for the specified service (OpenAI or Anthropic)
    still has a valid access token.

    API tokens are valid for 4 hours (14400 seconds)
    """
    now = time.time()
    if api_service not in ["oai", "anthropic"]:
        return True

    if not last_client_refresh or now - last_client_refresh[api_service] > TOKEN_VALIDITY_SECONDS:
        logger.info("Client has an expired token, a new one needs to be generated")
        return False
    return True



def get_client_api_key(api_service: str) -> str:
    """
    Requests a new API access token for the specified corporate account (OpenAI or Anthropic).

    This function uses client credentials (API_CLIENT_ID and API_CLIENT_SECRET)
    which must be retrieved from LastPass and set as as environment variables
    """

    client_id = os.getenv("API_CLIENT_ID", key_dict.get("API_CLIENT_ID", ""))
    client_secret = os.getenv("API_CLIENT_SECRET", key_dict.get("API_CLIENT_SECRET", ""))

    url = key_dict.get("API_URL", "")
    if not client_id or not client_secret or not url:
        raise RuntimeError("Corporate API client credentials are not configured")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    scope_map = {"oai": "azureopenai-readwrite", "anthropic": "awsanthropic-readwrite"}
    data = {"grant_type": "client_credentials", "scope": scope_map[api_service]}

    response = requests.post(url, headers=headers, data=data, auth=HTTPBasicAuth(client_id, client_secret))
    if response.status_code != 200:
        raise RuntimeError("Error: Could not generate a Bearer API token, please try again")

    return response.json()["access_token"]


def setup_oai_client() -> "OpenAI":
    """Set up corporate OpenAI client with an API key that needs to be refreshed every 4 hours"""
    from openai import OpenAI

    last_client_refresh["oai"] = time.time()
    oai_api_key = get_client_api_key("oai")

    # Initialize OpenAI client with custom base URL and headers
    oai_client = OpenAI(
        api_key=oai_api_key,
        base_url="https://prod.api.nvidia.com/llm/v1/azure",
        default_headers={"dataClassification": "sensitive", "dataSource": "internet"},
    )
    return oai_client


# SERVER_URL = None
if __name__ == "__main__":
    print(SERVER_URL)
