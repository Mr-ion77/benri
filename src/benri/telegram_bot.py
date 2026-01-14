import os
import sys
sys.modules['apscheduler'] = None
import subprocess
import re
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import pytz
import json
from data import aggregate_and_save_top_configs
import pandas as pd
import ast

# ======= GLOBAL CONSTANTS =======
MY_USER_ID = 6265691693
STATE_DIR = "/home/carlosR/QTransformer/ExperimentsForThesis/"
RESULTS_ROOT = "/home/carlosR/QTransformer_Results_and_Datasets/transformer_results/"

# --- Helpers ---

def get_screen_output(screen_id):
    """Dumps the current screen buffer to a file and reads it."""
    tmp_file = f"/tmp/{screen_id}_log.txt"
    # Tells screen to write the current view to a file
    subprocess.run(["screen", "-S", screen_id, "-X", "hardcopy", tmp_file])
    if os.path.exists(tmp_file):
        with open(tmp_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            print(f"DEBUG LOG CONTENT FOR SCREEN {screen_id}:\n{content}, \nLength: {len(content)}")  # Debug print
        return content
    return ""

def find_folder_by_number(short_id):
    """
    If short_id is a number (e.g. '8'), looks for a folder starting with '8_'.
    Otherwise, returns the short_id as is (assuming it's a full name).
    """
    if short_id.isdigit():
        try:
            # Look for a directory in STATE_DIR that starts with f"{short_id}_"
            for folder in os.listdir(STATE_DIR):
                if folder.startswith(f"{short_id}_") and os.path.isdir(os.path.join(STATE_DIR, folder)):
                    return folder
        except Exception:
            return short_id
    return short_id

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_USER_ID: return
    if not context.args:
        await update.message.reply_text("Usage: `/summary [number]`")
        return

    folder_name = find_folder_by_number(context.args[0]) # e.g., '8_Experiment_...'
    folder_path = os.path.join(STATE_DIR, folder_name)
    
    # 1. Find the script file: WIP_tests_*.py
    script_file = None
    for f in os.listdir(folder_path):
        if f.startswith("WIP_tests_") and f.endswith(".py"):
            script_file = os.path.join(folder_path, f)
            break
    
    if not script_file:
        await update.message.reply_text(f"❌ Could not find script `WIP_tests_*.py` in `{folder_name}`")
        return

    # 2. Extract variables from the script using Regex or ast
    try:
        with open(script_file, "r") as f:
            script_content = f.read()

        # Regex to find specific variables in the text
        exp_id_match = re.search(r"'experiment_id':\s*'([^']+)'", script_content)
        pixels_match = re.search(r"'pixels'\s*:\s*(\d+)", script_content)
        # Use literal_eval for the list to be safe
        graph_cols_match = re.search(r"graph_columns\s*=\s*(\[[^\]]+\])", script_content)

        if not (exp_id_match and graph_cols_match):
            await update.message.reply_text("❌ Could not parse `experiment_id` or `graph_columns` from script.")
            return

        exp_id_val = exp_id_match.group(1)
        pixels_val = pixels_match.group(1) if pixels_match else "Unknown"
        graph_columns = ast.literal_eval(graph_cols_match.group(1))

        # 3. Access the CSV Results
        csv_dir = os.path.join(RESULTS_ROOT, exp_id_val)
        csv_path = os.path.join(csv_dir, "results_grid_search.csv")

        if not os.path.exists(csv_path):
            await update.message.reply_text(f"❌ CSV not found at:\n`{csv_path}`", parse_mode="Markdown")
            return

        # 4. Process Data
        df = pd.read_csv(csv_path)
        
        # Run your aggregation function
        result_text, _ = aggregate_and_save_top_configs(
            df, 
            graph_columns[:-1], # group by everything except last
            graph_columns[-1],  # target is 'test_auc'
            table_dir=csv_dir
        )
        result_text = result_text[graph_columns[:-1] + ['median', 'std']].sort_values(by='median', ascending=False).to_string(index=False)

        # 5. Send results
        header = f"📊 *Summary for {folder_name}*\n📍 Exp ID: `{exp_id_val}`\n🖼 Pixels: `{pixels_val}`\n\n"
        # If text is too long for Telegram (max 4096), send first chunk
        if len(result_text) > 3500:
            result_text = result_text[:3500] + "\n... (truncated)"

        await update.message.reply_text(f"{header}```\n{result_text}\n```", parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error processing summary: `{str(e)}`", parse_mode="Markdown")

async def list_experiments(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_USER_ID:
        return

    # Determine the filter status (default to 'running' if no argument provided)
    filter_status = context.args[0].lower() if context.args else "running"
    
    try:
        found_experiments = []
        
        # Walk through the directory tree
        for root, dirs, files in os.walk(STATE_DIR):
            if "state.json" in files:
                folder_name = os.path.basename(root)
                file_path = os.path.join(root, "state.json")
                
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    
                    # Extract data and normalize status for comparison
                    current_status = data.get('status', 'unknown').lower()
                    idx = data.get('idx', '?')
                    total = data.get('total', '?')
                    
                    # Check if this experiment matches the requested filter
                    # We use 'in' to handle cases like "COMPLETED ✅" vs "completed"
                    if filter_status in current_status:
                        found_experiments.append(
                            f"• `{folder_name}`\n"
                            f"  └ {idx}/{total} | {data.get('status', 'running')}"
                        )
                except Exception:
                    # If we can't read the JSON, we only show it if specifically looking for errors
                    if filter_status == "error":
                        found_experiments.append(f"• `{folder_name}`\n  └ ⚠️ Error reading JSON")

        if not found_experiments:
            await update.message.reply_text(
                f"📂 No experiments found with status: `{filter_status}`"
            )
            return

        found_experiments.sort()
        
        header = f"🔬 *Thesis Experiments ({filter_status.upper()}):*\n\n"
        response = header + "\n\n".join(found_experiments)
        await update.message.reply_text(response, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ Error scanning directory: `{e}`")

# --- Command Handlers ---
async def progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_USER_ID: return
    if not context.args:
        await update.message.reply_text("Usage: `/progress [number or name]`", parse_mode="Markdown")
        return
    
    # Resolve the ID (handles '8' -> '8_Experiment_...')
    folder_name = find_folder_by_number(context.args[0])
    state_file = os.path.join(STATE_DIR, folder_name, "state.json")

    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            data = json.load(f)
        
        idx, total = int(data['idx']), int(data['total'])
        percent = min(100, int((idx / total) * 100))
        bar = "█" * (percent // 10) + "░" * (10 - (percent // 10))

        await update.message.reply_text(
            f"📊 *Progress:* `{folder_name}`\n"
            f"Step: `{idx}` / `{total}`\n"
            f"`{bar}` {percent}%\n"
            f"Status: *{data.get('status', 'running')}*",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(f"❌ Folder/File not found for: `{folder_name}`")

async def list_screens(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Security check
    if update.effective_user.id != MY_USER_ID:
        return

    try:
        # We use check=False because screen -ls returns code 1 if no screens exist
        result = subprocess.run(['screen', '-ls'], capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr # Screen sometimes prints list to stderr

        # Debug print in your terminal to see what the bot sees
        print(f"DEBUG SCREEN OUTPUT:\n{output}")

        if "No Sockets found" in output or not output.strip():
            await update.message.reply_text("📭 No active screen sessions found.")
            return

        # Improved Regex: 
        # It looks for a number, a dot, then captures everything until a space or tab.
        # Example: 12345.test_exp	(Detached) -> captures 'test_exp'
        screen_names = re.findall(r'\d+\.([^\s\t(]+)', output)
        
        if screen_names:
            # Remove duplicates just in case
            unique_screens = list(set(screen_names))
            response = "🖥️ *Active Screen Sessions:*\n\n"
            for name in unique_screens:
                response += f"• `{name}`\n"
            await update.message.reply_text(response, parse_mode="Markdown")
        else:
            # If regex fails, show the raw output so we can see the format
            await update.message.reply_text(f"Could not parse screens. Raw output:\n`{output}`", parse_mode="Markdown")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error listing screens: {e}")

async def start_exp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_USER_ID: return
    if len(context.args) < 1:
        await update.message.reply_text("Usage: `/start [folder_number]`")
        return
    
    # Resolve '8' to '8_Experiment_7_With_More_QVC'
    folder_name = find_folder_by_number(context.args[0])
    
    # Assumes the script name is inside that folder
    # You can customize the script name or pass it as a 2nd arg
    script_path = os.path.join(STATE_DIR, folder_name, "WIP_tests_Transformer.py")
    
    if os.path.exists(script_path):
        # Start screen session named exactly like the folder
        subprocess.run(["screen", "-dmS", folder_name, "-h", "10000", "python3", script_path])
        await update.message.reply_text(f"🚀 Started script in screen: `{folder_name}`")
    else:
        await update.message.reply_text(f"❌ Script not found at `{script_path}`")

async def kill_exp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_USER_ID: return
    if not context.args: return
    
    # Resolve the ID
    screen_id = find_folder_by_number(context.args[0])
    
    # Terminate the screen session
    subprocess.run(["screen", "-S", screen_id, "-X", "quit"])
    await update.message.reply_text(f"💀 Session `{screen_id}` terminated.")

# --- Main Entry Point ---

from telegram.ext import Application

if __name__ == "__main__":
    # We bypass the Builder and go straight to the Application class.
    # We explicitly tell it NOT to use a job_queue here.
    app = (
        Application.builder()
        .token("8369851856:AAGjGTo4349KUOB0FycE-sXGI1EOB3eLkxo")
        .job_queue(None) 
        .build()
    )

    # Note: If even the above fails, you can initialize it like this:
    # app = Application(bot=Bot("TOKEN"), update_queue=asyncio.Queue())
    # but let's try the builder with job_queue(None) again in this specific order.

    app.add_handler(CommandHandler("screens", list_screens))
    app.add_handler(CommandHandler("progress", progress))
    app.add_handler(CommandHandler("start", start_exp))
    app.add_handler(CommandHandler("kill", kill_exp))
    app.add_handler(CommandHandler("experiments", list_experiments))
    app.add_handler(CommandHandler("summary", summary))
    
    print("🚀 : Bot is listening...")
    app.run_polling()