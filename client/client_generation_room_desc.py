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
"""
MCP Client for Qwen3-VL-235B-A22B-Thinking Model

This client connects to MCP (Model Context Protocol) servers and uses the
Qwen3-VL-235B-A22B-Thinking model via OpenAI's chat completions API.

Key Differences from GPT-OSS Version:
====================================
1. API Format:
   - Uses: client.chat.completions.create() (OpenAI chat completions)
   - Instead of: client.responses.create() (GPT-OSS custom format)

2. Message Format:
   - Uses: messages list with roles (user/assistant/tool)
   - Instead of: input list with custom types

3. Tool Format:
   - Uses: OpenAI function calling format with nested structure
   - Tools: [{"type": "function", "function": {...}}]
   - Instead of: [{"type": "function", "name": ..., "parameters": ...}]

4. Response Format:
   - Extracts: response.choices[0].message.content
   - Tool calls: response.choices[0].message.tool_calls
   - Reasoning: response.choices[0].message.reasoning (Qwen-specific)
   - Instead of: response.output with various item types

5. Tool Results:
   - Uses: role="tool" messages with tool_call_id
   - Instead of: function_call_output type entries

Usage:
======
python client_oai_exec_qwen3_vl.py --input_text <text_file> --server_paths <server1.py> [<server2.py> ...]
python client_oai_exec_qwen3_vl.py --input_text <text_file> --input_images <img1.jpg> [<img2.png> ...] --server_paths <server1.py>
"""

import asyncio
import sys
import os
import json
import time
import re
import base64
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import AsyncExitStack
import subprocess
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import openai
from dotenv import load_dotenv
from log import save_chat_log_html

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Warning: PIL (Pillow) not available. Images will not be resized for logs.")

SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'server'))

load_dotenv()  # load environment variables from .env

def get_task_definition_text(room_desc: str):
    return f"""
Task: Generate a room based on the following room description: {room_desc}.

=== OVERVIEW ===

You will use tools to generate a complete, realistic scene. The process involves two main steps:
1. Generate the room layout
2. Place objects strategically in multiple stages

=== STEP 1: GENERATE ROOM LAYOUT ===

Generate the room layout first. Based on the description, generate EITHER a single room OR a multi-room layout (a house / apartment with multiple connected rooms) — whichever the description implies.

[IMPORTANT] If the description implies a whole home or several spaces (e.g. "apartment", "house", "two-bedroom", "a living room and a kitchen", "loft with ..."), generate ALL of those rooms in ONE layout call. If it describes a single space (e.g. "a kitchen", "a bedroom"), generate just one room. When calling the layout generation tool, state how many rooms and which room types to create. The layout generation returns scene requirements PER ROOM.

The layout generation will return scene requirements containing:
- Recommended objects to place
- Style information for the room
- Other important placement guidelines

=== STEP 2: PLACE OBJECTS ===

This is the core of your task. Follow the guidelines below carefully.

[MULTI-ROOM] If the layout has more than one room, decorate EVERY room — do not leave any room empty. Work room by room: complete the placement stages below for one room (calling place_objects_in_room / move_one_object_with_condition_in_room with that room's room_id), then move on to the next room, until all rooms are furnished. Use each room's own scene requirements.

--- 2.1 OBJECT PLACEMENT STRATEGY ---

Place objects in multiple stages following this priority order:

STAGE 1: Large, Key Furniture (Highest Priority)
- Place all major furniture first: cabinets, counters, appliances, tables, chairs
- These are the foundational elements of the room
- [CRITICAL] Think thoroughly about the spatial relationships between ALL objects before placing them
- Consider adjacency, facing directions, and functional groupings

STAGE 2: Object Combos and Functional Groups
- Place objects that form natural combinations with existing furniture:
  * Chairs with tables/desks
  * Small appliances near relevant counters
  * Functional groupings (e.g., coffee maker near mugs)
- This prevents furniture from seeming isolated and creates a cohesive scene

STAGE 3: Supporting Surface Objects
- On EVERY piece of furniture that can support objects, place 2+ meaningful items:
  * Kitchen counters: utensils, cutting boards, canisters, small appliances
  * Tables/desks: plates, bowls, food items, decorative objects
  * Shelves: dishes, glasses, cookbooks, jars (>5 items per shelf)
  * Cabinets with visible surfaces: storage containers, decorative items
- Do NOT place objects on furniture that cannot support them (e.g., lamps)

STAGE 4: Decorative and Background Objects
- Add remaining objects necessary for completeness:
  * Wall decorations (pictures, clocks)
  * Additional small decor items
  * Any other objects that enhance realism

--- 2.2 PLACEMENT SOURCES ---

Decide what objects to place by consulting THREE sources in order:

SOURCE 1: Scene Requirements (from layout generation)
- Follow the recommended objects list from the layout generation result
- This is your primary reference

SOURCE 2: Semantic Critic Info Recommendations
- Review the placement recommendations in semantic critic info
- [CRITICAL] Check for "stop_scene_generation": true — if present, the room is good enough, verify conditions and STOP
- Check "room_rating": if "excellent" or "good", focus on fixing specific issues, NOT adding more objects
- Check "analysis_reasoning" for detailed problem descriptions

--- MOVE/REPLACE/REMOVE actions (HIGHEST PRIORITY) ---
- [CRITICAL] If next_step actions contain Priority scores >= 7 for MOVE/REPLACE/REMOVE, you MUST fix these BEFORE adding any new objects
- Use move_one_object_with_condition_in_room(room_id, condition) for each MOVE action
- The condition format must follow: "Move [object_type] (object_id: [object_id]) to [floor|wall|object_id], [spatial_guidance]"
- Example: if critic says "Rotate chair to face desk", call move_one_object_with_condition_in_room with the chair's object_id

--- ADD actions (lower priority) ---
- Only add new objects AFTER all high-priority MOVE/REPLACE fixes are done
- Combine ADD actions into one place_objects_in_room call
- Do NOT over-add — respect the room_rating guidance

SOURCE 3: Failed Placements (Retry Logic)
- If object placement fails, RETRY ONCE with the same or similar objects
- If failures persist due to limited space, try smaller alternative objects
- Do NOT give up after one failure

--- 2.3 PLACEMENT CONSTRAINTS ---

PER-CALL LIMITS:
[CRITICAL] Place at most 6-8 objects and at most 5-6 object TYPES per single placement call.
[CRITICAL] Keep each tool call SHORT. Long placement calls cause the model to repeat
  the same object line over and over and fail to terminate — which silently aborts the
  whole generation. If a room needs more objects, make SEVERAL small calls instead of one big one.
[CRITICAL] NEVER repeat the same object entry within a single call.
[ENCOURAGED] Place a few related object types together per call, then call again for the rest.

TOTAL SCENE LIMITS:
- NO maximum on total objects across all calls
- NO maximum on total object types across all calls
- Add as many objects as needed for completeness and diversity

FORBIDDEN OBJECTS:
[CRITICAL] NEVER add: rugs, mats, curtains, blankets, ceiling-hanging objects (these are already installed)

REPLACEMENT WARNING:
[CRITICAL] NEVER use "replace all objects" - this removes everything you've placed

LIMIT HANDLING:
[CRITICAL] If you receive operation_type "limit_reached", do NOT stop immediately. The room may still need refinement. Use move_one_object_with_condition_in_room to fix orientations and positions. Use place_objects_in_room with REPLACE mode to swap problematic objects without increasing object count. Only stop when all orientations and placements look correct.

--- 2.4 STYLE CONSISTENCY ---

[IMPORTANT] All placed objects must match the style specified in the scene requirements
[IMPORTANT] Explicitly mention the object style when placing them

--- 2.5 OBJECT IDs ---

[IMPORTANT] Pay attention to place_id and object_id values
[IMPORTANT] Use exact object_id values when referencing existing objects for placement

--- 2.6 QUALITY REQUIREMENTS ---

Your scene MUST achieve the following:

✓ REALISM:
- Believable layout with correct positions, rotations, and sizes
- Common daily objects that make the room feel lived-in
- Rich with appropriate furniture and objects
- Objects of the same category are well-aligned
- Reasonable spatial relationships (e.g., chairs face tables)
- Sufficient walking space
- [CRITICAL] Correct orientations for all objects

✓ COMPLETENESS:
- Contains all necessary furniture for a kitchen (counters, cabinets, appliances, etc.)
- [CRITICAL] All necessary large AND small items are present
- [CRITICAL] Rich in details
- [CRITICAL] Every shelf is FULL of objects (>5 items per shelf minimum)
- [CRITICAL] Every supporter surface (tables, desks, shelves, counters) has small objects on it
- [CRITICAL] The room must feel finished and complete

✗ AVOID THESE MISTAKES:
- Unusual objects or strange placements that seem unrealistic
- Missing key objects or mismatched furniture for a kitchen
- Floating objects, crowded floors, abnormal sizes
- Object collisions or overlaps
- Incorrect orientations
- Large items placed oddly (e.g., refrigerator blocking walkway)
- Large empty spaces
- Blockers in front of functional furniture
- Sparse or empty rooms lacking decor or key elements

--- 2.7 WHEN TO STOP (COMPLETION CHECKLIST) ---

[CRITICAL] You MUST keep track of the current object placement status continuously.
[CRITICAL] Do NOT call get_current_layout to do this. Each place_objects_in_room result
  already tells you what was placed, and the layout-generation result already gave you every
  room's room_id and scene requirements. Calling get_current_layout dumps the entire
  (possibly multi-room) layout into the conversation, bloating context and making the model
  more likely to derail into a repetition loop. Track status from the tool results you
  already have instead.

DO NOT STOP until ALL of the following conditions are met (OR stop_scene_generation is signaled):

□ CONDITION 0: [HIGHEST PRIORITY] If semantic critic returns "stop_scene_generation": true, the room quality is sufficient — verify and STOP
□ CONDITION 1: All necessary large AND small items are present with rich details
□ CONDITION 2: Every shelf is full of objects (>5 items per shelf)
□ CONDITION 3: Every supporter surface (tables, desks, shelves, counters) has small objects on it
□ CONDITION 4: All scene requirements from layout generation are satisfied
□ CONDITION 5: All semantic critic modification recommendations (MOVE/REPLACE/REMOVE) have been addressed

BEFORE STOPPING:
[CRITICAL] Explicitly verify each condition above
[CRITICAL] If any condition is not met, continue placing objects
[CRITICAL] If placements fail for small objects on surfaces, keep trying with different object types
[CRITICAL] Do NOT be afraid of placement failures - keep trying with alternatives
[CRITICAL] Do NOT stop prematurely - persistence is required

[ENCOURAGED] When in doubt, place more objects if they are reasonable additions

=== PLACEMENT LOCATION SPECIFICATION ===

When placing objects, you must specify the location using ONE of three types:

--- TYPE 1: "floor" ---

Use "floor" for objects that sit on the ground/floor surface.

This includes:
- Objects against walls
- Objects in corners  
- Objects beside/next to/left of/right of/in front of/behind other floor objects
- Any object whose base rests on the floor

Examples:
✓ "A table against the wall" → use "floor" (table sits on floor, near wall)
✓ "A sofa in the corner" → use "floor" (sofa sits on floor, in corner)
✓ "A chair next to the table" → use "floor" (chair sits on floor, near table)
✓ "A chair in front of the desk" → use "floor" (chair sits on floor, facing desk)
✓ "A cabinet beside the refrigerator" → use "floor" (cabinet sits on floor, next to fridge)

--- TYPE 2: "wall" ---

Use "wall" ONLY for objects that are:
- Physically attached or mounted directly ON the wall surface
- Positioned above the floor level

Examples:
✓ "A picture on the wall" → use "wall" (mounted on wall surface, above floor)
✓ "A wall-mounted shelf" → use "wall" (attached to wall, above floor)
✓ "A wall-mounted TV" → use "wall" (mounted on wall, above floor)
✓ "A wall cabinet above the counter" → use "wall" (attached to wall, above floor)

--- TYPE 3: "object_id" or "estimated_object_name" ---

Use object references ONLY for objects placed onto/on top of another existing object.

When to use each:
- Use "object_id" when you know the exact ID of the existing object (preferred)
- Use "estimated_object_name" when you don't know the exact ID or the object is about to be placed

Examples:
✓ "A lamp on a table" → use the table's object_id (lamp goes on top of table)
✓ "A plate on the kitchen counter" → use the counter's object_id (plate goes on top)
✓ "A vase on the cabinet" → use the cabinet's object_id (vase goes on top)
✓ "Books on the shelf" → use the shelf's object_id (books go inside/on the shelf)

--- CRITICAL DISTINCTIONS ---

[CRITICAL] The key distinction is the spatial relationship:
- "on top of" → use object_id or estimated_object_name
- "beside", "next to", "in front of", "behind", "left of", "right of" → use "floor"

[CRITICAL] Be precise - do NOT use "floor" or "wall" as a lazy fallback when the location should be an object_id

[CRITICAL] When referencing existing objects, use the exact object_id whenever possible

--- PLACEMENT FORMAT ---

Specify placements as:
"place [quantity] [object_type] [description] on [location (floor|wall|object_id)], [extra guidance]"

Example:
"place 3 kitchen knife modern style on object_id_42, aligned in a row"
"place 1 dining chair wooden on floor, facing the table"
"place 1 wall clock round on wall, above the doorway"

[IMPORTANT] In tool returns, the placement description `place on [place_location]` follows this same pattern.

"""

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_and_encode_image(image_path: str, max_size: tuple = (512, 512)) -> tuple:
    """Resize image and encode to base64 for logging
    
    Args:
        image_path: Path to image file
        max_size: Maximum dimensions (width, height) for resized image
    
    Returns:
        tuple: (base64_string, original_size, resized_size, file_size)
    """
    if not PIL_AVAILABLE:
        # Fallback: just encode original image
        base64_str = encode_image_to_base64(image_path)
        file_size = os.path.getsize(image_path)
        return base64_str, (0, 0), (0, 0), file_size
    
    try:
        # Open and get original size
        with Image.open(image_path) as img:
            original_size = img.size
            file_size = os.path.getsize(image_path)
            
            # Convert to RGB if necessary (for PNG with transparency, etc.)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            resized_size = img.size
            
            # Encode resized image to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            base64_str = base64.b64encode(buffer.read()).decode('utf-8')
            
            return base64_str, original_size, resized_size, file_size
            
    except Exception as e:
        print(f"⚠️  Failed to resize image {image_path}: {e}")
        # Fallback to original
        base64_str = encode_image_to_base64(image_path)
        file_size = os.path.getsize(image_path)
        return base64_str, (0, 0), (0, 0), file_size

def get_image_mime_type(image_path: str) -> str:
    """Get MIME type based on file extension"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'image/jpeg')

class MCPClientOAI:
    """
    MCP Client for Qwen3-VL-235B-A22B-Thinking model
    
    This client uses the OpenAI chat completions API format to interact with
    the Qwen3-VL thinking model. Key features:
    - Uses chat.completions.create() instead of responses.create()
    - Supports tool calling in OpenAI format
    - Captures and displays model reasoning (Qwen thinking model feature)
    - Maintains conversation history in messages format (user/assistant/tool roles)
    - Tracks token usage and tool call limits
    """
    def __init__(self):
        # Initialize session and client objects - now supporting multiple servers
        self.servers: Dict[str, Dict[str, Any]] = {}  # server_name -> {session, stdio, write, tools}
        self.exit_stack = AsyncExitStack()

        self.layout_id = None
        
        # Initialize OpenAI client with Qwen3-VL endpoint
        from env_config import load_client_config
        key_dict = load_client_config()

        API_TOKEN = key_dict["API_TOKEN"]
        API_URL_QWEN = key_dict["API_URL_QWEN"]
        self.MODEL_NAME = key_dict["MODEL_NAME"]

        # Initialize OpenAI client with Qwen3-VL endpoint
        self.openai_client = openai.OpenAI(
            base_url=API_URL_QWEN,
            api_key=API_TOKEN
        )
        
        # Use messages list format for chat completions
        self.messages: List[Dict[str, Any]] = []
        
        # Initialize logging
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.session_start_time = datetime.now()
        self.log_filename = self._generate_log_filename()
        
        # Initialize token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        
        # Initialize tool call tracking (raised from old hard-coded 15 to allow
        # richer scenes; configurable via SAGE_MAX_TOOL_CALLS).
        self.tool_call_count = 0
        self.max_tool_calls = int(os.environ.get("SAGE_MAX_TOOL_CALLS", "40"))
        # Older tool results are compressed to a short head to bound context.
        self.full_tool_result_window = int(os.environ.get("SAGE_FULL_TOOL_RESULT_WINDOW", "6"))
        self.tool_result_truncate_chars = int(os.environ.get("SAGE_TOOL_RESULT_TRUNCATE_CHARS", "600"))

    def _generate_log_filename(self) -> str:
        """Generate a timestamped log filename"""
        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        return f"chat_session_qwen3vl_{timestamp}.json"

    def _save_chat_log(self):
        """Save the current conversation history to a log file"""
        try:
            # Organize conversation by role and type for better readability
            conversation_summary = {
                "user_messages": 0,
                "user_images": 0,
                "assistant_messages": 0,
                "function_calls": 0,
                "function_outputs": 0,
                "reasoning_entries": 0
            }
            
            organized_conversation = []
            
            for msg in self.messages:
                # Handle both dictionary and object types
                if hasattr(msg, 'get'):
                    # It's a dictionary
                    msg_type = msg.get('role', msg.get('type', 'unknown'))
                elif hasattr(msg, 'type'):
                    # It's an object with type attribute
                    msg_type = msg.type
                elif hasattr(msg, 'role'):
                    # It's an object with role attribute
                    msg_type = msg.role
                else:
                    msg_type = 'unknown'
                
                if msg_type == 'user':
                    conversation_summary["user_messages"] += 1
                    content = msg.get('content', '') if hasattr(msg, 'get') else getattr(msg, 'content', '')
                    timestamp = msg.get('timestamp', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'timestamp', 'unknown')
                    
                    # Count and extract images if present
                    images_data = []
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'image_url':
                                conversation_summary["user_images"] += 1
                                # Get image metadata if stored
                                if 'image_metadata' in item:
                                    images_data.append(item['image_metadata'])
                    
                    # Check for images_metadata field (legacy format)
                    images_metadata = msg.get('images_metadata', []) if hasattr(msg, 'get') else []
                    if images_metadata:
                        images_data.extend(images_metadata)
                    
                    entry = {
                        "type": "user_message",
                        "content": content,
                        "timestamp": timestamp
                    }
                    if images_data:
                        entry["images"] = images_data
                    
                    organized_conversation.append(entry)
                elif msg_type == 'assistant':
                    conversation_summary["assistant_messages"] += 1
                    if hasattr(msg, 'get'):
                        # Dictionary format
                        entry = {
                            "type": "assistant_response",
                            "content": msg.get('content', ''),
                            "iteration": msg.get('iteration', 'unknown'),
                            "timestamp": msg.get('timestamp', 'unknown'),
                            "has_tool_calls": msg.get('has_tool_calls', False)
                        }
                        if 'reasoning' in msg and msg['reasoning']:
                            entry["reasoning"] = msg['reasoning']
                            conversation_summary["reasoning_entries"] += 1
                        if 'tool_calls' in msg and msg['tool_calls']:
                            conversation_summary["function_calls"] += len(msg['tool_calls'])
                    else:
                        # Object format
                        entry = {
                            "type": "assistant_response",
                            "content": getattr(msg, 'content', ''),
                            "iteration": getattr(msg, 'iteration', 'unknown'),
                            "timestamp": getattr(msg, 'timestamp', 'unknown'),
                            "has_tool_calls": getattr(msg, 'has_tool_calls', False)
                        }
                        if hasattr(msg, 'reasoning') and msg.reasoning:
                            entry["reasoning"] = msg.reasoning
                            conversation_summary["reasoning_entries"] += 1
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            conversation_summary["function_calls"] += len(msg.tool_calls)
                    organized_conversation.append(entry)
                elif msg_type == 'function_call':
                    conversation_summary["function_calls"] += 1
                    # Handle ResponseFunctionToolCall objects
                    organized_conversation.append({
                        "type": "function_call",
                        "name": getattr(msg, 'name', 'unknown'),
                        "arguments": getattr(msg, 'arguments', ''),
                        "call_id": getattr(msg, 'call_id', 'unknown')
                    })
                elif msg_type == 'tool':
                    # New format for tool results (chat completions format)
                    conversation_summary["function_outputs"] += 1
                    organized_conversation.append({
                        "type": "tool_result",
                        "tool_name": msg.get('tool_name', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'tool_name', 'unknown'),
                        "server": msg.get('server', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'server', 'unknown'),
                        "content": msg.get('content', '') if hasattr(msg, 'get') else getattr(msg, 'content', ''),
                        "tool_call_id": msg.get('tool_call_id', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'tool_call_id', 'unknown'),
                        "timestamp": msg.get('timestamp', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'timestamp', 'unknown'),
                        "error": msg.get('error', False) if hasattr(msg, 'get') else getattr(msg, 'error', False)
                    })
                elif msg_type == 'function_call_output':
                    # Legacy format (kept for backwards compatibility)
                    conversation_summary["function_outputs"] += 1
                    if hasattr(msg, 'get'):
                        # Dictionary format
                        organized_conversation.append({
                            "type": "function_output",
                            "tool_name": msg.get('tool_name', 'unknown'),
                            "server": msg.get('server', 'unknown'),
                            "output": msg.get('output', ''),
                            "call_id": msg.get('call_id', 'unknown'),
                            "timestamp": msg.get('timestamp', 'unknown'),
                            "error": msg.get('error', False)
                        })
                    else:
                        # Object format
                        organized_conversation.append({
                            "type": "function_output",
                            "tool_name": getattr(msg, 'tool_name', 'unknown'),
                            "server": getattr(msg, 'server', 'unknown'),
                            "output": getattr(msg, 'output', ''),
                            "call_id": getattr(msg, 'call_id', 'unknown'),
                            "timestamp": getattr(msg, 'timestamp', 'unknown'),
                            "error": getattr(msg, 'error', False)
                        })
                elif msg_type == 'reasoning':
                    conversation_summary["reasoning_entries"] += 1
                    # Handle ResponseReasoningItem objects
                    content = getattr(msg, 'content', [])
                    if isinstance(content, list):
                        reasoning_texts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                reasoning_texts.append(item['text'])
                            else:
                                reasoning_texts.append(str(item))
                        content_str = ' | '.join(reasoning_texts)
                    else:
                        content_str = str(content)
                    
                    organized_conversation.append({
                        "type": "reasoning",
                        "id": getattr(msg, 'id', 'unknown'),
                        "content": content_str,
                        "raw_content": content
                    })
            
            log_data = {
                "session_start": self.session_start_time.isoformat(),
                "session_end": datetime.now().isoformat(),
                "connected_servers": {
                    name: info['path'] for name, info in self.servers.items()
                },
                "conversation_summary": conversation_summary,
                "organized_conversation": organized_conversation,
                "input_list": [self._serialize_message(msg) for msg in self.messages],  # Keep for HTML compatibility
                "total_messages": len(self.messages),
                "client_type": "qwen3vl_chat_completions",
                "token_usage": {
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                    "total_tokens": self.total_tokens,
                    "api_call_count": self.api_call_count,
                    "average_tokens_per_call": self.total_tokens / max(1, self.api_call_count)
                },
                "tool_usage": {
                    "total_tool_calls": self.tool_call_count,
                    "max_tool_calls": self.max_tool_calls,
                    "limit_reached": self.tool_call_count >= self.max_tool_calls
                }
            }
            
            log_path = self.logs_dir / self.log_filename
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Enhanced chat log saved to: {log_path}")
            print(f"📊 Token usage: {self.total_input_tokens} input + {self.total_output_tokens} output = {self.total_tokens} total tokens")
            print(f"🔧 Tool usage: {self.tool_call_count}/{self.max_tool_calls} tool calls")
            print(f"💬 Conversation summary:")
            print(f"   👤 User messages: {conversation_summary['user_messages']}")
            if conversation_summary['user_images'] > 0:
                print(f"   📸 User images: {conversation_summary['user_images']}")
            print(f"   🤖 Assistant messages: {conversation_summary['assistant_messages']}")
            print(f"   🧠 Reasoning entries: {conversation_summary['reasoning_entries']}")
            print(f"   🔧 Tool calls: {conversation_summary['function_calls']}")
            print(f"   📤 Tool results: {conversation_summary['function_outputs']}")

            save_chat_log_html(str(log_path))

            print(f"💾 Chat log saved to: {log_path.with_suffix('.html')}")
            
        except Exception as e:
            print(f"⚠️  Failed to save chat log: {str(e)}")

    def _serialize_message(self, msg):
        """Convert message objects to serializable dictionaries"""
        if hasattr(msg, 'get'):
            # Already a dictionary - but might need enhancement for HTML compatibility
            
            # Remove content_for_api from serialization (only keep resized images in 'content')
            if 'content_for_api' in msg:
                msg_copy = msg.copy()
                msg_copy.pop('content_for_api', None)
                msg = msg_copy
            
            if msg.get('role') == 'assistant' and 'reasoning' in msg:
                # Convert reasoning to format expected by HTML generator
                enhanced_msg = msg.copy()
                reasoning_content = msg['reasoning']
                
                # Add reasoning as string items in the input list (for HTML compatibility)
                reasoning_items = []
                for reasoning_item in reasoning_content:
                    reasoning_str = str(reasoning_item)
                    # Format as expected by HTML generator
                    if 'reasoning_text' in reasoning_str.lower() or 'text' in str(reasoning_item):
                        reasoning_items.append(f"ResponseReasoningItem: 'text': '{reasoning_str}', 'type': 'reasoning_text'")
                    else:
                        reasoning_items.append(f"ResponseReasoningItem: {reasoning_str}")
                
                # Keep original reasoning for our enhanced view
                enhanced_msg['reasoning_items'] = reasoning_items
                return enhanced_msg
            return msg
        elif hasattr(msg, '__dict__'):
            # Object with attributes - convert to dict
            result = {}
            msg_type = getattr(msg, 'type', getattr(msg, 'role', 'unknown'))
            
            # Handle function call objects specially
            if msg_type == 'function_call' or hasattr(msg, 'name') and hasattr(msg, 'arguments'):
                # Format as string for HTML compatibility
                name = getattr(msg, 'name', 'unknown')
                arguments = getattr(msg, 'arguments', '{}')
                call_id = getattr(msg, 'call_id', 'unknown')
                return f"ResponseFunctionToolCall: name='{name}', arguments='{arguments}', call_id='{call_id}'"
            
            # Handle other objects
            for attr in dir(msg):
                if not attr.startswith('_'):  # Skip private attributes
                    try:
                        value = getattr(msg, attr)
                        # Only include serializable types
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            result[attr] = value
                        elif callable(value):
                            continue  # Skip methods
                        else:
                            result[attr] = str(value)
                    except:
                        continue
            
            # Add type information if available
            if hasattr(msg, 'type'):
                result['type'] = msg.type
            elif hasattr(msg, 'role'):
                result['role'] = msg.role
                
            return result
        else:
            # Fallback - convert to string
            return {"raw_content": str(msg), "type": "unknown"}

    def _auto_save_log(self):
        """Auto-save the chat log periodically"""
        # Save every 10 messages to prevent data loss
        if len(self.messages) > 0:
            self._save_chat_log()

    async def connect_to_servers(self, server_script_paths: List[str]):
        """Connect to multiple MCP servers

        Args:
            server_script_paths: List of paths to server scripts (.py or .js)
        """
        
        for i, server_script_path in enumerate(server_script_paths):
            server_name = f"server_{i+1}"  # Give each server a unique name
            print(f"\n🔌 Connecting to {server_name}: {server_script_path}")
            
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                print(f"⚠️  Skipping {server_script_path}: Server script must be a .py or .js file")
                continue

            try:
                if is_python:
                    # Run the layout server with the same interpreter as the client by
                    # default.  This keeps local conda runs portable and also allows the
                    # client to run inside a container that has no conda executable.
                    import os
                    import shlex
                    abs_script_path = os.path.abspath(server_script_path)
                    server_python = os.environ.get("SAGE_SERVER_PYTHON", sys.executable)
                    
                    # Forward ALL SAGE_* tuning vars to the layout server. Without this
                    # the MCP server (started via conda run + bash -c) never sees knobs
                    # like SAGE_CRITIC_FREQUENCY / SAGE_SEMANTIC_VLM_VIEWS / SAGE_PREVIEW_*,
                    # so they are silently ignored.
                    sage_exports = "".join(
                        f"export {k}={shlex.quote(v)} && "
                        for k, v in os.environ.items()
                        if k.startswith("SAGE_") and v is not None
                    )

                    # Create a bash command that forwards runtime tuning and starts the
                    # stdio MCP server.  CONDA_PREFIX is optional in container images.
                    server_command = (
                        f"cd {SERVER_DIR} && "
                        f"export LD_LIBRARY_PATH=${{CONDA_PREFIX:+$CONDA_PREFIX/lib:}}${{LD_LIBRARY_PATH:-}} && "
                        f"export LIBRARY_PATH=${{CONDA_PREFIX:+$CONDA_PREFIX/lib:}}${{LIBRARY_PATH:-}} && "
                        f"export CPATH=${{CONDA_PREFIX:+$CONDA_PREFIX/include:}}${{CPATH:-}} && "
                        f"export PKG_CONFIG_PATH=${{CONDA_PREFIX:+$CONDA_PREFIX/lib/pkgconfig:}}${{PKG_CONFIG_PATH:-}} && "
                        f"export SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')} && "
                        f"export PHYSICS_CRITIC_ENABLED={os.environ.get('PHYSICS_CRITIC_ENABLED', 'true')} && "
                        f"export SEMANTIC_CRITIC_ENABLED={os.environ.get('SEMANTIC_CRITIC_ENABLED', 'true')} && "
                        f"{sage_exports}"
                        f"{shlex.quote(server_python)} {shlex.quote(abs_script_path)}"
                    )
                    
                    command = "bash"
                    args = ["-c", server_command]
                else:
                    # Use node for JavaScript scripts
                    command = "node"
                    args = [server_script_path]
                
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=os.environ.copy(),
                )

                print(f"🔌 Connecting to {server_name} with command: {command} {args}")

                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                
                # Print transport information (stdio uses pipes, not ports)
                print(f"📡 Transport type: stdio (standard input/output pipes)")
                print(f"📡 Server process command: {command}")
                print(f"📡 Server process args: {args}")
                
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

                print(f"🔌 Connected to {server_name} (stdio communication established)")

                await session.initialize()

                print(f"🔌 Initializing {server_name}")

                # List available tools for this server
                response = await session.list_tools()
                tools = response.tools
                
                # Store server info
                self.servers[server_name] = {
                    'session': session,
                    'stdio': stdio,
                    'write': write,
                    'tools': tools,
                    'path': server_script_path
                }
                
                print(f"✅ Connected to {server_name} with {len(tools)} tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
                    
            except Exception as e:
                print(f"❌ Failed to connect to {server_script_path}: {str(e)}")
                continue

        if not self.servers:
            raise Exception("Failed to connect to any servers")
        
        total_tools = sum(len(server_info['tools']) for server_info in self.servers.values())
        print(f"\n🎉 Successfully connected to {len(self.servers)} servers with {total_tools} total tools")

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all connected servers in OpenAI chat completions format"""
        all_tools = []
        tool_server_map = {}  # tool_name -> server_name (for routing)
        
        for server_name, server_info in self.servers.items():
            for tool in server_info['tools']:
                # Handle potential tool name conflicts by prefixing with server name
                original_name = tool.name
                tool_name = tool.name
                
                # If tool name already exists, prefix with server name
                if tool_name in tool_server_map:
                    tool_name = f"{server_name}_{original_name}"
                    # Also update the previous conflicting tool
                    conflicting_server = tool_server_map[original_name]
                    # Find and update the conflicting tool in all_tools
                    for existing_tool in all_tools:
                        if existing_tool['function']['name'] == original_name:
                            existing_tool['function']['name'] = f"{conflicting_server}_{original_name}"
                            break
                    tool_server_map[f"{conflicting_server}_{original_name}"] = conflicting_server
                    del tool_server_map[original_name]
                
                tool_server_map[tool_name] = server_name
                
                # Convert MCP tool format to OpenAI chat completions format
                all_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"[{server_name}] {tool.description}",
                        "parameters": tool.inputSchema
                    }
                })
        
        # Store the mapping for tool routing
        self.tool_server_map = tool_server_map
        return all_tools

    async def call_tool_on_server(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Route tool call to the correct server"""
        # Find which server this tool belongs to
        server_name = self.tool_server_map.get(tool_name)
        if not server_name:
            raise Exception(f"Tool {tool_name} not found on any server")
        
        # Extract original tool name (remove server prefix if present)
        original_tool_name = tool_name
        if tool_name.startswith(f"{server_name}_"):
            original_tool_name = tool_name[len(f"{server_name}_"):]
        
        server_info = self.servers[server_name]
        session = server_info['session']
        
        return await session.call_tool(original_tool_name, tool_input)
    
    def parse_tool_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse <tool_call> blocks from Qwen's text output
        
        Qwen sometimes formats tool calls as:
        <tool_call>
        {"name": "tool_name", "arguments": {...}}
        </tool_call>
        
        This method extracts and parses these into a list of tool call dictionaries.
        """
        tool_calls = []
        
        # Find all <tool_call>...</tool_call> blocks
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for i, match in enumerate(matches):
            try:
                json_str = match.group(1).strip()
                tool_data = json.loads(json_str)
                
                # Create a tool call object similar to OpenAI's format
                tool_call = {
                    'id': f'call_parsed_{i}_{int(time.time() * 1000)}',
                    'type': 'function',
                    'function': {
                        'name': tool_data.get('name', ''),
                        'arguments': json.dumps(tool_data.get('arguments', {}))
                    }
                }
                tool_calls.append(tool_call)
                print(f"🔍 Parsed tool call from content: {tool_data.get('name', 'unknown')}")
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse tool call JSON: {e}")
                print(f"   JSON string was: {json_str[:200]}...")
                continue
        
        return tool_calls

    async def process_query(self, query: str, image_paths: Optional[List[str]] = None) -> Tuple[str, bool]:
        """Process a query using Qwen3-VL and available tools from all servers
        
        Args:
            query: Text query/instruction
            image_paths: Optional list of image file paths to include in the query
        
        Returns:
            tuple: (final_response, intermediate_responses_shown)
        """
        
        # Build user message content (with images if provided)
        if image_paths:
            # Multi-modal content: text + images
            content_for_api = [{"type": "text", "text": query}]
            content_for_log = [{"type": "text", "text": query}]
            images_metadata = []
            
            for image_path in image_paths:
                try:
                    print(f"📸 Loading image: {image_path}")
                    
                    # Full-size image for API
                    base64_image_full = encode_image_to_base64(image_path)
                    mime_type = get_image_mime_type(image_path)
                    
                    content_for_api.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image_full}"
                        }
                    })
                    
                    # Resized image for log
                    base64_image_resized, original_size, resized_size, file_size = resize_and_encode_image(image_path)
                    
                    content_for_log.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_resized}"
                        },
                        "image_metadata": {
                            "filename": os.path.basename(image_path),
                            "path": image_path,
                            "mime_type": mime_type,
                            "original_size": {"width": original_size[0], "height": original_size[1]},
                            "resized_size": {"width": resized_size[0], "height": resized_size[1]},
                            "file_size_bytes": file_size,
                            "file_size_kb": round(file_size / 1024, 2),
                            "note": "Image resized to 512x512 max for log storage"
                        }
                    })
                    
                    images_metadata.append({
                        "filename": os.path.basename(image_path),
                        "path": image_path,
                        "mime_type": mime_type,
                        "original_size": {"width": original_size[0], "height": original_size[1]},
                        "resized_size": {"width": resized_size[0], "height": resized_size[1]},
                        "file_size_bytes": file_size,
                        "file_size_kb": round(file_size / 1024, 2)
                    })
                    
                    print(f"   ✅ Image loaded: {os.path.basename(image_path)} ({mime_type}, {round(file_size/1024, 2)} KB)")
                    if PIL_AVAILABLE and resized_size[0] > 0:
                        print(f"      Original: {original_size[0]}x{original_size[1]} → Resized for log: {resized_size[0]}x{resized_size[1]}")
                        
                except Exception as e:
                    print(f"   ⚠️  Failed to load image {image_path}: {e}")
            
            # Use content_for_log for storage (resized images)
            # But we'll store both for the API call
            user_message = {
                "role": "user",
                "content": content_for_log,  # Store resized version in log
                "content_for_api": content_for_api,  # Store full version for API (not serialized)
                "images_metadata": images_metadata,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Text-only content
            user_message = {
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            }
        
        # Add user message to messages list
        self.messages.append(user_message)
        self._auto_save_log()  # Auto-save after each user message

        # Get available tools from all servers
        available_tools = self.get_all_tools()

        # Main conversation loop (like Claude Desktop)
        max_iterations = 200  # Prevent infinite loops
        iteration = 0
        intermediate_responses_shown = False
        
        while iteration < max_iterations:
            iteration += 1
            
            print(f"\n🔄 Processing iteration {iteration}...")
            
            # Call Qwen3-VL with current messages and tools
            try:
                def build_messages_for_api():
                    # Compress older tool results (the main context-bloat source) so a
                    # high max_tool_calls stays within the context window. Rebuilt on
                    # each retry so emergency compaction (below) takes effect.
                    tool_msg_indices = [
                        i for i, m in enumerate(self.messages)
                        if isinstance(m, dict) and m.get('role') == 'tool'
                    ]
                    recent_tool_indices = set(tool_msg_indices[-self.full_tool_result_window:] if self.full_tool_result_window > 0 else [])

                    # Prepare messages for API (use full-size images if available)
                    built = []
                    for i, msg in enumerate(self.messages):
                        if isinstance(msg, dict) and 'content_for_api' in msg:
                            # Use full-size images for API call
                            api_msg = msg.copy()
                            api_msg['content'] = msg['content_for_api']
                            # Remove content_for_api and images_metadata from API message
                            api_msg.pop('content_for_api', None)
                            api_msg.pop('images_metadata', None)
                            built.append(api_msg)
                        elif (isinstance(msg, dict) and msg.get('role') == 'tool'
                              and i not in recent_tool_indices):
                            api_msg = msg.copy()
                            content = api_msg.get('content', '')
                            if isinstance(content, str) and len(content) > self.tool_result_truncate_chars:
                                api_msg['content'] = (
                                    content[:self.tool_result_truncate_chars]
                                    + f"\n...[older tool result truncated, {len(content)} chars total]"
                                )
                            built.append(api_msg)
                        else:
                            built.append(msg)
                    return built

                messages_for_api = build_messages_for_api()

                # Prepare tools parameter - don't pass tools at all if empty
                # Output budget. Placement calls are short now (6-8 objects per
                # call), so a 16k cap is generous while leaving far more of the
                # model context window for the conversation itself.
                try:
                    _max_tokens = int(os.environ.get("SAGE_MAX_TOKENS", "16384"))
                except ValueError:
                    _max_tokens = 16384
                call_params = {
                    "model": self.MODEL_NAME,
                    "messages": messages_for_api,
                    "max_tokens": _max_tokens,
                    "temperature": 1.0,
                }
                # Anti-repetition: the self-hosted Qwen3-VL-MoE can degenerate into a
                # repetition loop while emitting a long tool-call (e.g. repeating the
                # same "place 1 X ..." item until it hits max_tokens, leaving the
                # <tool_call> unterminated and unparseable). A frequency penalty breaks
                # this. DashScope didn't loop, and frequency_penalty is OpenAI-standard
                # so it's safe either way. Tunable / disable via env.
                try:
                    _freq_pen = float(os.environ.get("SAGE_FREQUENCY_PENALTY", "0.3"))
                except ValueError:
                    _freq_pen = 0.3
                if _freq_pen:
                    call_params["frequency_penalty"] = _freq_pen
                try:
                    _rep_pen = float(os.environ.get("SAGE_REPETITION_PENALTY", "1.0"))
                except ValueError:
                    _rep_pen = 1.0
                if _rep_pen and _rep_pen != 1.0:  # vLLM-specific; opt-in to stay API-portable
                    call_params["extra_body"] = {"repetition_penalty": _rep_pen}
                
                # Only add tools if we have any
                if available_tools:
                    call_params["tools"] = available_tools
                    call_params["tool_choice"] = "auto"  # Let the model decide when to use tools
                    print(f"🛠️  Available tools: {len(available_tools)}")
                
                # Retry mechanism with exponential backoff (for API errors)
                max_retry = 3
                last_error = None
                
                for retry in range(max_retry):
                    try:
                        if retry > 0:
                            wait_time = 2 ** retry  # Exponential backoff: 2, 4, 8 seconds
                            print(f"⏳ Retry attempt {retry + 1}/{max_retry} after {wait_time}s delay...", file=sys.stderr)
                            time.sleep(wait_time)
                        
                        response = self.openai_client.chat.completions.create(**call_params)
                        
                        # Track token usage if available in response
                        if hasattr(response, 'usage'):
                            usage = response.usage
                            input_tokens = getattr(usage, 'prompt_tokens', 0)
                            output_tokens = getattr(usage, 'completion_tokens', 0) 
                            total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens)
                            
                            self.total_input_tokens += input_tokens
                            self.total_output_tokens += output_tokens
                            self.total_tokens += total_tokens
                            self.api_call_count += 1
                            
                            print(f"🔢 Request processed successfully (tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total)")
                        else:
                            print(f"🔢 Request processed successfully")
                        
                        # If successful, break out of retry loop
                        break
                        
                    except Exception as e:
                        last_error = e
                        print(f"❌ API call failed: {str(e)}")
                        # Context-window overflow: don't just retry the same payload —
                        # aggressively compact the history (keep fewer full tool
                        # results, truncate older ones harder) and rebuild before the
                        # next attempt, so a long multi-room run degrades gracefully
                        # instead of dying.
                        err_text = str(e).lower()
                        if any(tok in err_text for tok in (
                            "context length", "maximum context", "context_length",
                            "too many tokens", "longer than the model", "max_model_len",
                        )):
                            self.full_tool_result_window = max(1, self.full_tool_result_window // 2)
                            self.tool_result_truncate_chars = max(200, self.tool_result_truncate_chars // 2)
                            print(
                                f"🗜️  Context overflow — compacting history "
                                f"(full_tool_result_window={self.full_tool_result_window}, "
                                f"tool_result_truncate_chars={self.tool_result_truncate_chars}) and retrying..."
                            )
                            messages_for_api = build_messages_for_api()
                            call_params["messages"] = messages_for_api
                        if retry == max_retry - 1:
                            # Final retry failed
                            error_msg = f"Error calling Qwen3-VL API after {max_retry} attempts: {str(e)}"
                            return error_msg, intermediate_responses_shown
                
            except Exception as e:
                error_msg = f"Error calling Qwen3-VL API: {str(e)}"
                return error_msg, intermediate_responses_shown

            # Process the chat completion response
            text_responses = []
            tool_calls = []
            reasoning_text = None
            
            # Extract the message from response
            try:
                message = response.choices[0].message
                
                # Extract text content
                if hasattr(message, 'content') and message.content:
                    text_responses.append(message.content)
                    print(f"📝 Response content: {len(message.content)} chars")
                
                # Extract reasoning if available (Qwen thinking model specific)
                if hasattr(message, 'reasoning') and message.reasoning:
                    reasoning_text = message.reasoning
                    print(f"🧠 Reasoning available: {len(reasoning_text)} chars")
                
                # Extract tool calls from API response
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls = message.tool_calls
                    print(f"🔧 Found {len(tool_calls)} tool call(s) in API response")
                
                # If no tool calls in API response, try parsing from content
                # (Qwen sometimes formats tool calls as <tool_call> XML tags in content)
                if not tool_calls and message.content and '<tool_call>' in message.content:
                    print(f"🔍 No tool calls in API response, parsing from content...")
                    parsed_tool_calls = self.parse_tool_calls_from_content(message.content)
                    if parsed_tool_calls:
                        # Convert parsed dict format to object-like format
                        class ToolCall:
                            def __init__(self, data):
                                self.id = data['id']
                                self.type = data['type']
                                self.function = type('Function', (), {
                                    'name': data['function']['name'],
                                    'arguments': data['function']['arguments']
                                })()
                        
                        tool_calls = [ToolCall(tc) for tc in parsed_tool_calls]
                        print(f"🔧 Parsed {len(tool_calls)} tool call(s) from content")
                    
            except Exception as e:
                print(f"⚠️  Error processing response: {e}")
                import traceback
                traceback.print_exc()
                # Fallback
                text_responses.append(f"Error processing response: {str(e)}")

            # Display reasoning if available (thinking process)
            if reasoning_text:
                print(f"\n🧠 Qwen3-VL Reasoning/Thinking:")
                reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                print(f"  💭 {reasoning_preview}")

            # Build assistant message for logging and history
            assistant_msg = {
                "role": "assistant",
                "content": message.content if hasattr(message, 'content') else "",
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "has_tool_calls": len(tool_calls) > 0
            }
            
            # Add tool calls if any
            if tool_calls:
                tool_calls = tool_calls[:1]
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            
            # Add reasoning if available
            if reasoning_text:
                assistant_msg["reasoning"] = reasoning_text
            
            # Add assistant message to messages list
            self.messages.append(assistant_msg)
            self._auto_save_log()

            # Display text responses
            if text_responses:
                full_text = "\n".join(text_responses)
                print(f"\n🤖 Qwen3-VL Response:")
                print(f"   📝 Content length: {len(full_text)} characters")
                print(f"   🔤 Response: {full_text}")
                intermediate_responses_shown = True

            # If no tool calls, we're done — UNLESS the model clearly *intended* a
            # tool call but it failed to parse (e.g. the self-hosted Qwen3-VL-MoE
            # degenerated into a repetition loop and hit max_tokens with an
            # unterminated/garbled <tool_call>). Silently returning there is the
            # "multi-room: no output" bug: the agent ends with 0 objects placed.
            # Instead, feed a corrective message back and let it retry — bounded by
            # a small retry counter so a persistently-broken model still terminates.
            if not tool_calls:
                content_str = message.content if hasattr(message, 'content') and message.content else ""
                looks_like_truncated_tool_call = '<tool_call>' in content_str
                self._malformed_tool_call_retries = getattr(self, '_malformed_tool_call_retries', 0)
                if looks_like_truncated_tool_call and self._malformed_tool_call_retries < 3:
                    self._malformed_tool_call_retries += 1
                    print(f"⚠️  Detected an unparseable/truncated <tool_call> (likely a "
                          f"repetition loop). Asking the model to retry "
                          f"({self._malformed_tool_call_retries}/3)...")
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Your previous tool call could not be parsed — it was "
                            "truncated or contained repeated/garbled content. Re-issue "
                            "a SINGLE, well-formed tool call now. Keep it concise: "
                            "place at most 5-8 objects in one place_objects_in_room "
                            "call, never repeat the same object line, and make sure the "
                            "JSON arguments are complete and valid. Do NOT call "
                            "get_current_layout."
                        ),
                        "timestamp": datetime.now().isoformat(),
                    })
                    continue
                final_response = "\n".join(text_responses) if text_responses else "No response from Qwen3-VL."
                print(f"✅ Conversation complete after {iteration} iteration(s)")
                return final_response, intermediate_responses_shown
            # A successful tool call resets the malformed-retry budget.
            self._malformed_tool_call_retries = 0
            
            # Check if we've reached the tool call limit
            if self.tool_call_count >= self.max_tool_calls:
                warning_msg = f"⚠️  Reached maximum tool call limit ({self.max_tool_calls}). Stopping execution."
                print(f"\n{warning_msg}")
                final_response = "\n".join(text_responses) if text_responses else warning_msg
                return final_response, intermediate_responses_shown

            # Execute tool calls and collect results
            print(f"\n🔧 Executing {len(tool_calls)} tool call(s)...")
            for i, tool_call in enumerate(tool_calls):
                # Check if we've reached the limit before executing this tool
                if self.tool_call_count >= self.max_tool_calls:
                    warning_msg = f"Reached maximum tool call limit ({self.max_tool_calls}). Skipping remaining {len(tool_calls) - i} tool call(s)."
                    print(f"\n  ⚠️  {warning_msg}")
                    # Add error result for skipped tool calls
                    tool_result_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": warning_msg,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.messages.append(tool_result_msg)
                    continue
                
                try:
                    tool_name = tool_call.function.name
                    tool_arguments = tool_call.function.arguments
                    server_name = self.tool_server_map.get(tool_name, "unknown")
                    
                    print(f"\n  🔧 Tool {i+1}/{len(tool_calls)}: {tool_name} on {server_name}")
                    print(f"     📥 Arguments: {tool_arguments}")
                    
                    # Increment tool call counter
                    self.tool_call_count += 1
                    print(f"     📊 Tool call count: {self.tool_call_count}/{self.max_tool_calls}")
                    
                    # Parse the arguments JSON
                    tool_input = json.loads(tool_arguments)
                    result = await self.call_tool_on_server(tool_name, tool_input)
                    
                    # Handle different result types including images
                    if hasattr(result, 'content'):
                        if isinstance(result.content, list):
                            # Multiple content blocks (text, images, etc.)
                            content_blocks = []
                            for item in result.content:
                                if hasattr(item, 'type'):
                                    if item.type == 'text' and hasattr(item, 'text'):
                                        content_blocks.append(item.text)
                                    elif item.type == 'image':
                                        content_blocks.append(f"[Image: {getattr(item.source, 'media_type', 'unknown')}]")
                                        print(f"     📸 Image result: {getattr(item.source, 'media_type', 'unknown')}")
                                    else:
                                        content_blocks.append(str(item))
                                else:
                                    content_blocks.append(str(item))
                            
                            result_content = "\n".join(content_blocks)
                            print(f"     📤 Result: {result_content[:200]}{'...' if len(result_content) > 200 else ''}")
                        else:
                            # Single content block
                            result_content = str(result.content)
                            print(f"     📤 Result: {result_content[:200]}{'...' if len(result_content) > 200 else ''}")
                    else:
                        result_content = str(result)
                        print(f"     📤 Result: {result_content[:200]}{'...' if len(result_content) > 200 else ''}")

                    result_content_dict = json.loads(result_content)

                    if 'layout_id' in result_content_dict:
                        self.layout_id = result_content_dict['layout_id']

                    
                    # Add tool result to messages list in chat completions format
                    tool_result_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content,
                        "timestamp": datetime.now().isoformat(),
                        "tool_name": tool_name,
                        "server": server_name
                    }
                    self.messages.append(tool_result_msg)
                    
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    print(f"     ❌ Error: {error_msg}")
                    
                    # Add error result to messages list
                    error_result_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat(),
                        "tool_name": tool_name,
                        "server": server_name,
                        "error": True
                    }
                    self.messages.append(error_result_msg)

            self._auto_save_log()  # Auto-save after each tool execution

            # Continue the loop to get OpenAI's final response
            print(f"🔄 Continuing to next iteration...")
            print(f"🔍 Layout ID: {self.layout_id}")
            continue

        return "Conversation exceeded maximum iterations. Please try a simpler query.", intermediate_responses_shown

    async def cleanup(self):
        """Clean up resources"""
        self._save_chat_log()  # Save final log on exit
        await self.exit_stack.aclose()

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='MCP Client for Qwen3-VL-235B-A22B-Thinking model with multi-modal support',
    )
    
    parser.add_argument(
        '--room_desc',
        type=str,
        required=True,
        help='Room description'
    )
    
    parser.add_argument(
        '--input_images',
        type=str,
        nargs='*',
        default=None,
        help='Optional list of image file paths to include with the query (supports jpg, png, gif, webp, bmp)'
    )
    
    parser.add_argument(
        '--server_paths',
        type=str,
        nargs='+',
        required=True,
        help='List of MCP server script paths (.py files)'
    )

    parser.add_argument(
        '--max_tool_calls',
        type=int,
        default=int(os.environ.get("SAGE_MAX_TOOL_CALLS", "40")),
        help='Maximum number of MCP tool calls before stopping (default 40 or $SAGE_MAX_TOOL_CALLS)'
    )
    
    args = parser.parse_args()
    
    # Read the instruction from the file
    room_desc = args.room_desc

    text_instruction = get_task_definition_text(room_desc)

    task_description_dict = {
        "room_desc": room_desc,
        "task_definition_text": text_instruction,
        "finished": False
    }
    
    room_desc_head = room_desc.strip(" \n\t").replace(' ', '_').replace('.', '_').replace('/', '_')[:20]
    label_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{room_desc_head}"
    save_dir = "./room_descs"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{label_id}.json"), "w") as f:
        json.dump(task_description_dict, f, indent=4)

    if not text_instruction:
        print(f"❌ Error: Instruction is empty: {args.input_text}")
        sys.exit(1)
    
    # Validate image paths if provided
    image_paths = None
    if args.input_images:
        image_paths = []
        for img_path in args.input_images:
            if not os.path.exists(img_path):
                print(f"⚠️  Warning: Image file not found: {img_path}")
            else:
                image_paths.append(img_path)
        
        if not image_paths:
            print(f"⚠️  No valid image paths found, proceeding with text-only query")
            image_paths = None
        else:
            print(f"📸 Found {len(image_paths)} image(s) to include")
    
    client = MCPClientOAI()
    client.max_tool_calls = args.max_tool_calls
    
    try:
        print(f"🔌 Connecting to {len(args.server_paths)} MCP server(s)...")
        await client.connect_to_servers(args.server_paths)
        print(f"🔌 Connected to {len(args.server_paths)} MCP server(s)")
        
        # Execute the single instruction and exit
        print(f"\n📝 Executing instruction: {text_instruction[:100]}{'...' if len(text_instruction) > 100 else ''}")
        if image_paths:
            print(f"📸 Including {len(image_paths)} image(s) in the query")
        print("\n🤔 Qwen3-VL is thinking...")
        
        response, responses_shown = await client.process_query(text_instruction, image_paths)

        # Print the final response if no intermediate responses were shown
        if response and not responses_shown:
            print(f"\n🤖 Qwen3-VL: {response}")

        # Deterministic final step (do not rely on the agent calling it): export the
        # integrated whole-scene result — combined USD collection + whole-layout
        # preview renders — so multi-room layouts read as one building.
        if "export_full_scene" in client.tool_server_map:
            try:
                print("\n🏠 Exporting integrated whole-scene (USD collection + full preview)...")
                export_result = await client.call_tool_on_server("export_full_scene", {})
                print(f"🏠 Whole-scene export result: {getattr(export_result, 'content', export_result)}")
            except Exception as e:
                print(f"⚠️  Whole-scene export failed (non-fatal): {e}")

        print(f"\n✅ Instruction completed. Exiting...")

        print(f"🔍 Layout ID: {client.layout_id}")

        task_description_dict["finished"] = True
        task_description_dict["layout_id"] = client.layout_id
        with open(os.path.join(save_dir, f"{label_id}.json"), "w") as f:
            json.dump(task_description_dict, f, indent=4)
        
        print("Task info saved to: ", os.path.join(save_dir, f"{label_id}.json"))
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
