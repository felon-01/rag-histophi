
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from rag_chain import retrieve, format_chat_history
    print("✅ Imports successful.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test History Formatting
history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
formatted = format_chat_history(history)
assert "User: Hi" in formatted
assert "Assistant: Hello" in formatted
print("✅ Chat history formatting works.")

print("✅ Basic smoke test passed. (Skipping model inference to avoid large downloads in agent environment)")
