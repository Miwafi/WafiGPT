import os, json, torch, shutil
from safetensors.torch import load_file
from train import *
from collections import OrderedDict
from colorama import init as colorama_init, Fore, Style, Back

colorama_init(autoreset=True)

THINK_COLOR = Fore.BLACK + Back.WHITE
RESET = Style.RESET_ALL
INDENT = "  "

def get_terminal_width():
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80

def print_wrapped(text, indent=INDENT, width=None):
    if width is None:
        width = get_terminal_width()
    available_width = width - len(indent)
    if available_width <= 0:
        available_width = 40
    
    lines = []
    current_line = ""
    for char in text:
        if len(current_line) >= available_width:
            lines.append(current_line)
            current_line = char
        else:
            current_line += char
    if current_line:
        lines.append(current_line)
    
    for i, line in enumerate(lines):
        if i == 0:
            print(f"{indent}{line}", end="")
        else:
            print(f"\n{indent}{line}", end="")

# ================================================

def find_latest_model(model_base_dir="./model"):
    if not os.path.exists(model_base_dir):
        raise FileNotFoundError(f"Model directory not found: {model_base_dir}")
    
    best_model_path = os.path.join(model_base_dir, "best_model")
    if os.path.exists(best_model_path) and os.path.exists(os.path.join(best_model_path, "model.safetensors")):
        return best_model_path
    
    model_dirs = []
    for item in os.listdir(model_base_dir):
        item_path = os.path.join(model_base_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "model.safetensors")):
            model_dirs.append(item_path)
    
    if not model_dirs:
        raise FileNotFoundError(f"No valid models found in {model_base_dir}")
    
    latest_model = max(model_dirs, key=lambda p: os.path.getmtime(p))
    return latest_model

# ================================================

def sample_next_token(logits, generated_tokens, repetition_penalty, presence_penalty, temperature):
    for token in set(generated_tokens):
        if logits[token] < 0:
            logits[token] *= repetition_penalty
        else:
            logits[token] /= repetition_penalty
    vocab_size = logits.size(0)
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    mask[list(set(generated_tokens))] = True
    logits[mask] += presence_penalty
    probs = torch.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item(), probs

# ================================================

def generate_response(model, tokenizer, prompt, device, config, max_length=512, temperature=0.3, repetition_penalty=1.0, presence_penalty=-1.5):
    encoded = tokenizer(f"<|user|>{prompt}<|assistant|>", update=False)
    generated = encoded["input_ids"].unsqueeze(0).to(device)
    unknown_id = tokenizer.split_tokens.get("<|unknown|>")
    end_id = tokenizer.split_tokens.get("<|end|>")
    newline_id = tokenizer.split_tokens.get("\\n")
    
    think_start_id = tokenizer.split_tokens.get("<|think|>")
    think_end_id = tokenizer.split_tokens.get("<|/think|>")
    
    in_thinking = False
    text_buffer = ""
    first_think = True

    with torch.no_grad():
        for _ in range(max_length):
            if generated.size(1) > config["max_seq_length"]:
                current_input = generated[:, -config["max_seq_length"] :]
                pos_offset = generated.size(1) - config["max_seq_length"]
            else:
                current_input = generated
                pos_offset = 0

            outputs = model(current_input, pos_offset=pos_offset)
            logits = outputs["logits"][0, -1, :].clone()
            gen_tokens = generated[0].tolist()
            token_id, probs = sample_next_token(logits, gen_tokens, repetition_penalty, presence_penalty, temperature)

            if token_id == unknown_id and probs.sum() > 0:
                probs[unknown_id] = 0.0
                probs = probs / probs.sum()
                token_id = torch.multinomial(probs, num_samples=1).item()

            generated = torch.cat((generated, torch.tensor([[token_id]], device=generated.device)), dim=1)
            if token_id == end_id:
                break
            
            token_str = tokenizer.decode([token_id])
            text_buffer += token_str
            
            if token_id == think_start_id:
                in_thinking = True
                if first_think:
                    print(f"\n{INDENT}{THINK_COLOR} thinking {RESET}")
                    first_think = False
                else:
                    print(f"\n{INDENT}{THINK_COLOR} thinking {RESET}")
                text_buffer = ""
            elif token_id == think_end_id:
                in_thinking = False
                print(f"\n{INDENT}{Fore.BLACK}{Back.GREEN} Output {RESET}\n")
                text_buffer = ""
            elif in_thinking:
                if "</think>" in text_buffer:
                    parts = text_buffer.split("</think>")
                    content = parts[0]
                    if content:
                        for line in content.split("\n"):
                            if line.strip():
                                print(f"{INDENT}{Fore.WHITE}{line}{RESET}")
                    print(f"\n{INDENT}{Fore.BLACK}{Back.GREEN} Output {RESET}\n")
                    in_thinking = False
                    text_buffer = parts[1] if len(parts) > 1 else ""
                    if text_buffer:
                        print_wrapped(text_buffer, indent=INDENT)
                        text_buffer = ""
                elif token_id == newline_id:
                    line = text_buffer[:-1].strip()
                    if line:
                        print(f"{INDENT}{Fore.WHITE}{line}{RESET}")
                    text_buffer = ""
                else:
                    pass
            else:
                if "<think>" in text_buffer:
                    parts = text_buffer.split("<think>")
                    before_think = parts[0]
                    if before_think:
                        print_wrapped(before_think, indent=INDENT)
                    in_thinking = True
                    if first_think:
                        print(f"\n{INDENT}{THINK_COLOR} thinking {RESET}")
                        first_think = False
                    else:
                        print(f"\n{INDENT}{THINK_COLOR} thinking {RESET}")
                    text_buffer = parts[1] if len(parts) > 1 else ""
                elif token_id == newline_id:
                    print_wrapped(text_buffer[:-1], indent=INDENT)
                    print()
                    text_buffer = ""
                elif len(text_buffer) > 10:
                    print_wrapped(text_buffer, indent=INDENT)
                    text_buffer = ""
    if text_buffer:
        if in_thinking:
            for line in text_buffer.split("\n"):
                if line.strip():
                    print(f"{INDENT}{Fore.WHITE}{line}{RESET}")
        else:
            print_wrapped(text_buffer, indent=INDENT)
    if not in_thinking:
        print()

# ================================================

def load_chat_model(model_dir, device):
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(model_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        token_dict = json.load(f)
    tokenizer = ChatTokenizer(config)
    tokenizer.split_tokens = OrderedDict(token_dict)
    model = ChatModel(config).to(device)
    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} | Device: {device}")
    return model, tokenizer, config

# ================================================

if __name__ == "__main__":
    print(f"\n  {Fore.BLACK}{Back.WHITE} WaFiGPT {RESET}\n")
    model_dir = find_latest_model()
    print(f"  Loading model from: {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_chat_model(model_dir, device)
    print()
    while True:
        try:
            prompt = input(f"  {Fore.CYAN}>{RESET} ")
            if prompt.strip().lower() in ["exit", "quit"]:
                break
            if prompt.strip():
                generate_response(model, tokenizer, prompt, device, config)
        except KeyboardInterrupt:
            print("\n")
            break
        except EOFError:
            print("\n")
            break
