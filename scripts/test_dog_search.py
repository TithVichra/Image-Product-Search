import os
import urllib.request
import json

dog_path = os.path.join(os.path.dirname(__file__), "dog_test.jpg")
print(f"Testing Dog Image: {dog_path} ({os.path.getsize(dog_path)} bytes)")

boundary = "----DogBoundary12345"
fn = "dog_test.jpg"
with open(dog_path, "rb") as f:
    file_bytes = f.read()

# 1. Test Detect endpoint
body_detect = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="{fn}"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req_det = urllib.request.Request(
    "http://127.0.0.1:8000/api/detect",
    data=body_detect,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)
resp_det = urllib.request.urlopen(req_det)
res_det = json.loads(resp_det.read().decode("utf-8"))
print("\n=== API DETECT RESULT ===")
print("is_out_of_domain:", res_det.get("is_out_of_domain"))
print("detected_entity:", res_det.get("detected_entity"))
print("out_of_domain_reason:", res_det.get("out_of_domain_reason"))
print("detected_boxes:", res_det.get("detected_boxes"))

# 2. Test Image Search endpoint
body_search = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="{fn}"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode("utf-8") + file_bytes + (
    f"\r\n--{boundary}\r\n"
    f'Content-Disposition: form-data; name="use_crop"\r\n\r\n'
    f"true\r\n"
    f"--{boundary}--\r\n"
).encode("utf-8")

req_search = urllib.request.Request(
    "http://127.0.0.1:8000/api/search/image",
    data=body_search,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)
resp_search = urllib.request.urlopen(req_search)
res_search = json.loads(resp_search.read().decode("utf-8"))
print("\n=== API SEARCH IMAGE RESULT ===")
print("is_out_of_domain:", res_search.get("is_out_of_domain"))
print("detected_entity:", res_search.get("detected_entity"))
print("out_of_domain_reason:", res_search.get("out_of_domain_reason"))
print("results count:", len(res_search.get("results", [])))
print("noise_metadata:", res_search.get("noise_metadata"))
