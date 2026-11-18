from dashboard.callbacks import update_fantasy
print("Testing update_fantasy...")
try:
    res = update_fantasy(1, 3)
    print("Success! Output type:", type(res))
except Exception as e:
    import traceback
    traceback.print_exc()
