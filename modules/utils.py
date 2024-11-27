import numpy as np
import cv2
import base64
from langchain_core.tools import BaseTool

import json

def construct_tools_json(tools: list[BaseTool]) -> str:

    tools_dict = {'tools': []}

    for tool in tools:
        tool_dict = {'name': tool.model_fields['name'].default,
                     'description': tool.model_fields['description'].default,
                     'parameters': {'properties': list(), 'required': list()}}
        for param_name, param_info in tool.model_fields['args_schema'].default.model_fields.items():
            tool_dict['parameters']['properties'].append({'name': param_name,
                                                          'description': param_info.description,
                                                          'type': param_info.annotation.__class__.__name__})
            if param_info.is_required():
                tool_dict['parameters']['required'].append(param_name)

        tools_dict['tools'].append(tool_dict)

    return json.dumps(tools_dict)


def img2b64(img: np.ndarray[np.uint8]) -> str:
    retval, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


def b642img(b64: str) -> np.ndarray[np.uint8]:
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)
