# Planning



## Inference
Planning is a real-time interactive task, which requires the model to generate a plan while the agent is executing the plan. As a result, we provide a server and a client for real-time planning which can be connected between the agent and the model on two different machines.


### Server
Run the server by run the following command
```bash
python server.py --model_path ${PATH_TO_MODEL_PATH}
```

### Client
Run the client by run the following command

```bash
python client.py --server_url ${SERVER_URL}
```

https://github.com/user-attachments/assets/60d93a3a-f964-4e41-9491-138a75595fbb

Users can register their own model in `models` folder and modify the parameter `LLM_TYPE_OPTIONS` in `client.py` to use their own model.

## Finetune

The finetune of planning is following the finetune of Rynnbrain base model. The only difference is that the data should be in the format of streaming for long-term memory.

Example of streaming data:
```json
{
    "id": , 
    "conversation": 
    [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": },  # main task prompt    
                {"type": "image", "image": , "width": , "height": }
            ]
        }, 
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": }, 
            ]
        }, 
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": , "width":, "height":},
                {"type": "image", "image": , "width":, "height":}
            ]
        },

        ......

        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": "Done."}, # End the planning with 'Done.'
            ]
        }, 
    ]
}

```
                