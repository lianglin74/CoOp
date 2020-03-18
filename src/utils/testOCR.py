import concurrent.futures

import requests

def testOCR():
    file_name = "./test.jpg"
    output_file_name = "./testOCR.txt"
    with open(file_name, "rb") as f:
        input_file = f.read()
        print("input got")

    headers = {
        'Content-Type':'image/jpg'
    }
    r = requests.post('http://vigdgx01:5001/vision/v2.3/read/core/Analyze', headers=headers, data=input_file)

    if (r.ok):
        with open(output_file_name, 'w', encoding='utf8') as f:
            print("Output: " + output_file_name)
            f.write(r.text)


if __name__ == "__main__":
    testOCR()
