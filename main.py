import os
from app_single import app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or fallback to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)
