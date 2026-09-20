import urllib.request
import json

ball_path = 'scripts/ball_test.jpg'
boundary = '----BallBoundary12345'
with open(ball_path, 'rb') as f:
    file_bytes = f.read()

body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="ball.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request('http://127.0.0.1:8000/api/detect', data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
res = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
print('=== DETECT RESULT FOR BALL ===')
print('is_out_of_domain:', res.get('is_out_of_domain'))
print('detected_boxes:', res.get('detected_boxes'))

body_search = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="ball.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + file_bytes + (
    f'\r\n--{boundary}\r\n'
    f'Content-Disposition: form-data; name="use_crop"\r\n\r\n'
    f'true\r\n'
    f'--{boundary}--\r\n'
).encode('utf-8')

req2 = urllib.request.Request('http://127.0.0.1:8000/api/search/image', data=body_search, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
res2 = json.loads(urllib.request.urlopen(req2).read().decode('utf-8'))
print('\n=== SEARCH RESULT FOR BALL ===')
print('is_out_of_domain:', res2.get('is_out_of_domain'))
print('results count:', len(res2.get('results', [])))
for r in res2.get('results', [])[:5]:
    print(f"Rank #{r['rank']}: {r['display_name']} | score: {r['score']} | img: {r['image_score']}")
