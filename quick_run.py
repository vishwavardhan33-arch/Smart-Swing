from main import SmartSwing

app = SmartSwing()
result = app.analyze_swing('golf_swing.mp4', session_name='Test Session')

if result:
    print(f"✅ Success! Score: {result['score']:.1f}/100")
else:
    print("❌ Analysis failed")
