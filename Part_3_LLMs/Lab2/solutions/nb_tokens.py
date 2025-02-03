total_characters = len(book)
total_tokens = len(tokenizer.encode(book))

print(f"Total number of characters in the book: {total_characters}")
print(f"Total number of tokens in the book: {total_tokens}")