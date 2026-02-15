import os
from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

# Pobierz klucze
pub = os.getenv("LANGFUSE_PUBLIC_KEY")
sec = os.getenv("LANGFUSE_SECRET_KEY")

print(f"Public key: {pub[:10]}... (długość: {len(pub) if pub else 0})")
print(f"Secret key: {sec[:10]}... (długość: {len(sec) if sec else 0})")

# Test połączenia
lf = Langfuse(public_key=pub, secret_key=sec, host="https://cloud.langfuse.com")
print("✅ Langfuse połączony!")

# Stwórz testowy trace
trace = lf.trace(name="TEST_Z_PYTHONA", input="test 123")
print(f"✅ Trace utworzony! ID: {trace.id}")

trace.generation(name="test_gen", input="hello").end(output="world")
print("✅ Generation dodany!")

lf.flush()
print("✅ Flush wykonany!")
print("\n🎯 Teraz wejdź na https://cloud.langfuse.com → Traces")
print(f"🔍 Szukaj trace o nazwie: TEST_Z_PYTHONA")