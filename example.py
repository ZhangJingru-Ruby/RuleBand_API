from ruleband_api import RuleBandAPI
import os, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 🩹 allow double OpenMP
warnings.filterwarnings("ignore", category=UserWarning, message=".*libiomp5md")

api = RuleBandAPI(device="cpu")
x, y = api.predict_from_file("data/example_data.json", debug=True)
print(f"Sampled rule-based sub-goal → ({x:.2f}, {y:.2f})")
