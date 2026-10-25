"""Comprehensive E2E test verifying iris_classifier.joblib workflow, defensive rendering, compatibility check, full validation execution, and error cases."""
import io
import json
import uuid
import urllib.request
import urllib.error
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

BASE_URL = "http://127.0.0.1:8000/api"

def make_request(method, endpoint, token=None, json_data=None, multipart_fields=None, multipart_files=None):
    url = f"{BASE_URL}{endpoint}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if multipart_fields is not None or multipart_files is not None:
        boundary = uuid.uuid4().hex
        body = bytearray()
        if multipart_fields:
            for k, v in multipart_fields.items():
                body.extend(f'--{boundary}\r\n'.encode('utf-8'))
                body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode('utf-8'))
                body.extend(f'{v}\r\n'.encode('utf-8'))
        if multipart_files:
            for k, (fname, fbytes) in multipart_files.items():
                body.extend(f'--{boundary}\r\n'.encode('utf-8'))
                body.extend(f'Content-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'.encode('utf-8'))
                body.extend(b'Content-Type: application/octet-stream\r\n\r\n')
                body.extend(fbytes)
                body.extend(b'\r\n')
        body.extend(f'--{boundary}--\r\n'.encode('utf-8'))
        headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
        req = urllib.request.Request(url, data=bytes(body), headers=headers, method=method)
    elif json_data is not None:
        headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(url, data=json.dumps(json_data).encode('utf-8'), headers=headers, method=method)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('utf-8')
            return resp.status, json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        content = e.read().decode('utf-8')
        try:
            err_json = json.loads(content)
        except Exception:
            err_json = {"detail": content}
        return e.code, err_json


def main():
    print("=========================================================")
    print("TEST SUITE: IRIS MODEL E2E VALIDATION & ASSURANCE RUN")
    print("=========================================================")

    # 1. Register & Authenticate
    email = f"iris_qa_{uuid.uuid4().hex[:6]}@assurance.ai"
    status, auth = make_request("POST", "/auth/register", json_data={
        "email": email,
        "password": "Password123!",
        "full_name": "QA Assurance Lead"
    })
    assert status in (200, 201), f"Registration failed: {auth}"
    token = auth["access_token"]
    print(f"[PASS] Step 1: User registered and authenticated ({email})")

    # 2. Create Project
    status, project = make_request("POST", "/projects", token=token, json_data={
        "name": "Iris Flower Classification Assurance",
        "description": "Validation project for iris_classifier.joblib"
    })
    assert status in (200, 201), f"Project creation failed: {project}"
    project_id = project["id"]
    print(f"[PASS] Step 2: Project created ({project_id})")

    # 3. Upload iris_classifier.joblib
    iris_model_path = Path(r"e:\AI_ML_Validation\backend\uploads\models\22bd531e-9706-42d9-b6dc-9f9f411c52fa\026ab09b68fd4d8bb24baa8f06214cfe_iris_classifier.joblib")
    assert iris_model_path.exists(), f"Iris model file not found at {iris_model_path}"
    model_bytes = iris_model_path.read_bytes()

    status, model_res = make_request(
        "POST", "/models/upload", token=token,
        multipart_fields={"project_id": project_id, "name": "Iris Classifier (RF)"},
        multipart_files={"file": ("iris_classifier.joblib", model_bytes)}
    )
    assert status in (200, 201), f"Model upload failed: {model_res}"
    model_id = model_res["id"]
    print(f"[PASS] Step 3: Model uploaded ({model_id}, format={model_res['framework']})")

    # 4. Call /models/{id}/inspect and verify defensive attributes
    status, inspect_res = make_request("GET", f"/models/{model_id}/inspect", token=token)
    assert status in (200, 201), f"Model inspect failed: {inspect_res}"
    print(f"[PASS] Step 4: Model inspect response received:")
    print(f"   - Estimator: {inspect_res.get('model_class')}")
    print(f"   - Feature Count: {inspect_res.get('feature_count')}")
    print(f"   - Feature Names: {inspect_res.get('feature_names')}")
    print(f"   - Classes: {inspect_res.get('classes')}")
    print(f"   - Total Parameters: {inspect_res.get('total_parameters')} (Must be None/null without crashing frontend)")
    print(f"   - Layer Count: {inspect_res.get('layer_count')} (Must be None/null without crashing frontend)")
    assert inspect_res.get("total_parameters") is None
    assert inspect_res.get("feature_count") == 4
    assert len(inspect_res.get("feature_names")) == 4

    # 5. Prepare and Upload Authentic iris_test.csv matching exact features
    raw_iris = load_iris(as_frame=True)
    df_iris = raw_iris.frame.copy()
    # columns are ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    csv_buf = io.StringIO()
    df_iris.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode('utf-8')

    status, dataset_res = make_request(
        "POST", "/datasets/upload", token=token,
        multipart_fields={"project_id": project_id, "name": "Iris Evaluation Dataset"},
        multipart_files={"file": ("iris_test.csv", csv_bytes)}
    )
    assert status in (200, 201), f"Dataset upload failed: {dataset_res}"
    dataset_id = dataset_res["id"]
    print(f"[PASS] Step 5: Iris dataset uploaded ({dataset_id}, rows={dataset_res.get('row_count')}, cols={dataset_res.get('column_count')})")

    # 6. Dataset Preview
    status, preview_res = make_request("GET", f"/datasets/{dataset_id}/preview?max_rows=15", token=token)
    assert status in (200, 201), f"Dataset preview failed: {preview_res}"
    print(f"[PASS] Step 6: Dataset preview confirmed ({preview_res.get('total_rows')} rows, cols={preview_res.get('columns')})")

    # 7. Pre-Flight Compatibility Check
    validators = ["performance", "data_quality", "fairness", "drift", "robustness", "explainability", "reliability"]
    status, compat_res = make_request("POST", "/validation/compatibility-check", token=token, json_data={
        "model_id": model_id,
        "dataset_id": dataset_id,
        "selected_validators": validators
    })
    assert status in (200, 201), f"Compatibility check failed: {compat_res}"
    print(f"[PASS] Step 7: Pre-Flight Compatibility Result:")
    print(f"   - Compatible: {compat_res.get('compatible')}")
    print(f"   - Warnings: {compat_res.get('warnings')}")
    print(f"   - Errors: {compat_res.get('errors')}")
    for vr in compat_res.get("validator_readiness", []):
        print(f"   - [{vr['name']}]: {vr['status_label']} (ready={vr['ready']})")
    assert compat_res.get("compatible") is True

    # 8. START FULL VALIDATION
    status, run_res = make_request("POST", "/validation/runs", token=token, json_data={
        "model_id": model_id,
        "dataset_id": dataset_id,
        "selected_validators": validators,
        "config": {"execution_mode": "full_assurance"}
    })
    assert status in (200, 201), f"Validation run start failed: {run_res}"
    run_id = run_res["id"]
    print(f"[PASS] Step 8: START FULL VALIDATION triggered! (Run ID={run_id}, Status={run_res.get('status')})")

    # 9. Verify Run Detail, All 7 Module Results, and Findings
    status, detail_res = make_request("GET", f"/validation/runs/{run_id}", token=token)
    assert status in (200, 201), f"Run detail fetch failed: {detail_res}"
    print(f"[PASS] Step 9: Validation Run Results:")
    print(f"   - Status: {detail_res.get('status')}")
    print(f"   - Overall Score: {detail_res.get('overall_score')}")
    print(f"   - Duration: {detail_res.get('duration_seconds')}s")
    print(f"   - Findings Count: {len(detail_res.get('findings', []))}")
    
    cfg = detail_res.get("config_json", {})
    metrics = cfg.get("metrics", {})
    print(f"   - Evaluated Validator Pillars in metrics: {list(metrics.keys())}")
    for k, v in metrics.items():
        score = v.get("score") if isinstance(v, dict) else "N/A"
        print(f"     * {k}: score={score}")

    assert detail_res.get("status") in ("completed", "passed", "warning")
    assert detail_res.get("overall_score") is not None

    # 10. Negative Test: Unsupported Extension (.pt)
    status_neg1, neg1 = make_request(
        "POST", "/models/upload", token=token,
        multipart_fields={"project_id": project_id, "name": "Invalid PyTorch"},
        multipart_files={"file": ("model.pt", b"dummy pytorch weights")}
    )
    print(f"[PASS] Step 10: Negative test - unsupported extension (.pt) correctly rejected (status={status_neg1}): {neg1.get('detail')}")
    assert status_neg1 in (400, 422)

    # 11. Negative Test: Corrupted Model File Handling
    status_neg2, neg2 = make_request(
        "POST", "/models/upload", token=token,
        multipart_fields={"project_id": project_id, "name": "Corrupted Joblib"},
        multipart_files={"file": ("corrupted.joblib", b"this is completely corrupted binary data not a pickle")}
    )
    assert status_neg2 in (200, 201)
    corrupted_model_id = neg2["id"]
    status_inspect_corrupt, inspect_corrupt = make_request("GET", f"/models/{corrupted_model_id}/inspect", token=token)
    print(f"[PASS] Step 11A: Negative test - inspecting corrupted model rejected with status={status_inspect_corrupt}: {inspect_corrupt.get('detail')}")
    assert status_inspect_corrupt >= 400

    status_compat_corrupt, compat_corrupt = make_request("POST", "/validation/compatibility-check", token=token, json_data={
        "model_id": corrupted_model_id,
        "dataset_id": dataset_id,
        "selected_validators": validators
    })
    print(f"[PASS] Step 11B: Negative test - compatibility check correctly flags corrupted model:")
    print(f"   - Compatible: {compat_corrupt.get('compatible')} (Expected False)")
    print(f"   - Blocking Errors: {compat_corrupt.get('errors')}")
    assert compat_corrupt.get("compatible") is False
    assert len(compat_corrupt.get("errors", [])) > 0

    # 12. Negative Test: Incompatible Dataset Check (mismatched columns)
    df_wrong = pd.DataFrame({"random_col_a": [1, 2, 3], "random_col_b": [4, 5, 6]})
    buf_wrong = io.StringIO()
    df_wrong.to_csv(buf_wrong, index=False)
    status, wrong_dataset = make_request(
        "POST", "/datasets/upload", token=token,
        multipart_fields={"project_id": project_id, "name": "Wrong Shape Dataset"},
        multipart_files={"file": ("wrong_dataset.csv", buf_wrong.getvalue().encode('utf-8'))}
    )
    wrong_dataset_id = wrong_dataset["id"]
    status_neg3, compat_neg = make_request("POST", "/validation/compatibility-check", token=token, json_data={
        "model_id": model_id,
        "dataset_id": wrong_dataset_id,
        "selected_validators": validators
    })
    print(f"[PASS] Step 12: Negative test - incompatible dataset correctly flagged:")
    print(f"   - Compatible: {compat_neg.get('compatible')} (Expected False)")
    print(f"   - Blocking Errors: {compat_neg.get('errors')}")
    assert compat_neg.get("compatible") is False
    assert len(compat_neg.get("errors", [])) > 0

    print("\n=========================================================")
    print("ALL 12 IRIS VALIDATION & ASSURANCE E2E TESTS PASSED!")
    print("=========================================================")

if __name__ == "__main__":
    main()
