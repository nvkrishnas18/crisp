from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Optional, Tuple
from uuid import UUID
from app.auth.auth import get_current_user
from app.agents import ds_agent
import json
from app.models.data_story import DataStory

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> Optional[Tuple[UUID, UUID, UUID, dict]]:
        try:
            token = websocket.query_params.get("token")
            story_uuid = websocket.query_params.get("story_uuid")
            is_reconnection = websocket.query_params.get("is_reconnection", "false").lower() == "true"

            if not token:
                await websocket.close(code=1008)  
                return None

            user = await get_current_user(token)
            if not user:
                print("Invalid token; unable to authenticate user.")
                await websocket.close(code=1008)
                return None
                
            if not story_uuid:
                print("No story_uuid provided.")
                await websocket.close(code=1008)
                return None

            data_story = await DataStory.find_one(DataStory.story_uuid == UUID(story_uuid))
            if not data_story:
                print(f"Data story with UUID {story_uuid} not found.")
                await websocket.close(code=1008)
                return None
            
            dataset_uuid = data_story.dataset_uuid
            print(f"Retrieved dataset_uuid {dataset_uuid} from story {story_uuid}")

            await websocket.accept()
            self.active_connections.append(websocket)

            print(f"Initializing agent with story_uuid: {story_uuid} and dataset_uuid: {dataset_uuid}")
            agent = ds_agent.get_agent(story_uuid=story_uuid, dataset_uuid=dataset_uuid)
            
            if is_reconnection:
                print("This is a reconnection - skipping initial summary")
                agent["skip_initial_summary"] = True
            else:
                print("This is a new connection - generating initial summary")
                has_summary = False
                if hasattr(data_story, "messages") and data_story.messages:
                    for msg in data_story.messages:
                        if msg.get("storage_type") == "initial_summary" and msg.get("version") == "v0":
                            has_summary = True
                            break
                
                if has_summary:
                    agent["skip_initial_summary"] = True
                    print("Found existing summary - won't generate a new one")
                else:
                    # agent = ds_agent.get_agent(story_uuid=story_uuid, dataset_uuid=dataset_uuid)
                    summary_msg = ds_agent.send_message(agent, "")
                    await websocket.send_json({"message": summary_msg})
            
            return user, dataset_uuid, story_uuid, agent

        except Exception as e:
            print(f"Error during WebSocket connection: {e}")
            import traceback
            traceback.print_exc()
            await websocket.close(code=1008)
            return None

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket, agent: dict):
        try:
            bot_msg = ds_agent.send_message(agent, message)
            await websocket.send_json({"message": bot_msg})
        except Exception as e:
            print(f"Error sending personal message: {e}")
            await websocket.send_json({"error": str(e)})

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    result = await manager.connect(websocket)
    if result:
        user, dataset_uuid, story_uuid, agent = result
        print(f"User info: {user}, DataSet UUID: {dataset_uuid}, Story UUID: {story_uuid}")
        try:
            while True:
                user_question = await websocket.receive_text()
                print(f"User question received: {user_question}")
                await manager.send_personal_message(user_question, websocket, agent)
        except WebSocketDisconnect:
            print(f"WebSocket disconnected for user {user}, dataset {dataset_uuid}, story {story_uuid}")
            manager.disconnect(websocket)
        except Exception as e:
            print(f"Error during WebSocket communication: {e}")
            import traceback
            traceback.print_exc()
            await websocket.close(code=1011)