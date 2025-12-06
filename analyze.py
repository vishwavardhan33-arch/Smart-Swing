import os
os.chdir('/Users/donivishwavardhan/Desktop/SmartSwing')

from main import SmartSwing

app = SmartSwing()
result = app.analyze_swing('golf_swing.mp4', session_name='Test')

if result:
    print(f"✅ Success! Score: {result['score']:.1f}/100")
