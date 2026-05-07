import os
import sys

def clear_screen() -> None:
    """
    Clears the terminal screen.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def print_data_table(data: dict) -> None:
    """
    Prints a clean table of market data.
    data: { symbol: { 'lp': price, 'close': close, 'change': change } }
    """
    # Move cursor to top left instead of full clear to reduce flicker
    sys.stdout.write("\033[H")

    print("="*50)
    print(f"{'SYMBOL':<15} | {'PRICE':<10} | {'CHANGE (%)':<10}")
    print("-"*50)

    for symbol, values in data.items():
        lp = values.get('lp', 0.0)
        close = values.get('close', 0.0)
        change = ((lp - close) / close * 100) if close else 0.0

        color = "\033[92m" if change >= 0 else "\033[91m"
        reset = "\033[0m"

        print(f"{symbol:<15} | {lp:<10.2f} | {color}{change:>8.2f}%{reset}")

    print("="*50)
    print("\nPress Ctrl+C to stop.")
    sys.stdout.flush()
