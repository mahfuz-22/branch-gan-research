# Token prediction test

from transformers import AutoTokenizer

def test_next_token_prediction(model, tokenizer, context):
    input_ids = tokenizer.encode(context, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generator.generator(input_ids)
    logits = outputs.logits
    predicted_token_id = torch.argmax(logits[0, -1, :]).item()
    predicted_token = tokenizer.decode([predicted_token_id])
    print(f"Context: {context}")
    print(f"Predicted next token: {predicted_token}")

# Use the function
context = "Fiction Context: then came the bats, and the star that was like ugh-lomi crept out of its blue hiding-place in the west. she called to"
test_next_token_prediction(trained_model, tokenizer, context)