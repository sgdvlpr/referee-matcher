import subprocess
import json

def ollama_chat(model_name, user_message):
    try:
        # Prepare the CLI command to send a message
        cmd = ['ollama', 'chat', model_name, '--prompt', user_message, '--json']

        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse the JSON output from ollama
        response = json.loads(result.stdout)
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')
    except Exception as e:
        print(f"Error calling Ollama CLI: {e}")
        return None

# Example usage
reply = ollama_chat('llama3', 'Hello from Python!')
print('Ollama replied:', reply)
