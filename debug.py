import requests
response = requests.get("https://newsapi.org/v2/everything?q=Tesla&apiKey=1949b3d75fa1494791dc4a3e9db37b07")
print(response.status_code)
print(response.json())

