import io

# Original line
# df = pd.read_csv(uploaded_file, encoding='utf-8-sig')

# Robust multi-encoding fallback logic
uploaded_file.seek(0)
raw = uploaded_file.read()
encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'cp1252', 'latin1']
last_err = None

for encoding in encodings:
    try:
        # Attempt to decode the bytes
        df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
        st.success(f'Successfully read file with encoding: {encoding}')
        break  # Break if successful
    except Exception as e:
        last_err = e
        continue
else:
        st.exception(last_err)
        st.stop()