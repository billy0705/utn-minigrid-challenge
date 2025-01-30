curl -X POST "http://localhost:8088/v1/chat/completions" \
-H "Content-Type: application/json" \
--data '{
"model": "Qwen/Qwen2.5-7B-Instruct",
"messages": [
{
"role": "user",
"content": "What is the capital of India?"
}
]
}'