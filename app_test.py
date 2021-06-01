import requests

resp = requests.post("https://localhost:5000/predict", files={"file": open('dataset/test_img/input.png','rb')})

print(resp)