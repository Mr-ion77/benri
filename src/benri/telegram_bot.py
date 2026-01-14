import os
import sys
sys.modules['apscheduler'] = None
import subprocess
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
import pytz
import json
from data import aggregate_and_save_top_configs
import pandas as pd
import ast

# ======= GLOBAL CONSTANTS =======
MY_USER_ID = 6265691693
STATE_DIR = "/home/carlosR/QTransformer/ExperimentsForThesis/"
RESULTS_ROOT = "/home/carlosR/QTransformer_Results_and_Datasets/"

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

def find_all_folders_by_number(short_id):
    """Returns a list of all folders starting with 'number_'."""
    matches = []
    if str(short_id).isdigit():
        try:
            for folder in os.listdir(STATE_DIR):
                if folder.startswith(f"{short_id}_") and os.path.isdir(os.path.join(STATE_DIR, folder)):
                    matches.append(folder)
        except Exception as e:
            print(f"Error listing folders: {e}")
    return sorted(matches)

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_USER_ID: return
    
    # This handler now supports both the command and button clicks
    query = update.callback_query
    if query:
        await query.answer()
        folder_name = query.data.replace("sum_", "")
    else:
        if not context.args:
            await update.message.reply_text("Usage: `/summary [number]`")
            return
        
        matches = find_all_folders_by_number(context.args[0])
        
        if len(matches) == 0:
            await update.message.reply_text(f"❌ No folders found starting with `{context.args[0]}_`")
            return
        
        if len(matches) > 1:
            # Create buttons for each match
            keyboard = [
                [InlineKeyboardButton(m, callback_data=f"sum_{m}")] for m in matches
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                f"🤔 Multiple experiments found for `{context.args[0]}`. Which one do you want?",
                reply_markup=reply_markup
            )
            return
        
        folder_name = matches[0]

    # --- Processing logic (same as before, using folder_name) ---
    await process_summary_logic(update, folder_name)

async def process_summary_logic(update, folder_name):
    # Determine if we are replying to a message or a callback query
    target = update.callback_query.message if update.callback_query else update.message
    
    folder_path = os.path.join(STATE_DIR, folder_name)
    
    # 1. Find the script file: WIP_tests_*.py
    script_file = next((f for f in os.listdir(folder_path) if f.startswith("WIP_tests_") and f.endswith(".py")), None)
    
    if not script_file:
        await target.reply_text(f"❌ No script found in `{folder_name}`")
        return

    try:
        script_full_path = os.path.join(folder_path, script_file)
        with open(script_full_path, "r") as f:
            content = f.read()

        # Extract values (Using the regex from our previous step)
        exp_id_val = re.search(r"'experiment_id':\s*'([^']+)'", content).group(1)
        graph_cols = ast.literal_eval(re.search(r"graph_columns\s*=\s*(\[[^\]]+\])", content).group(1))

        # 2. Path to CSV
        csv_path = f"/home/carlosR/QTransformer_Results_and_Datasets/{exp_id_val}/results_grid_search.csv"
        
        df = pd.read_csv(csv_path)
        group_cols = graph_cols[:-1]
        target_col = graph_cols[-1]

        summary_df = df.groupby(group_cols)[target_col].agg(['median', 'std']).reset_index()
        result_text = summary_df.sort_values(by='median', ascending=False).to_string(index=False)

        header = f"📊 *Summary:* `{folder_name}`\n"
        await target.reply_text(f"{header}```\n{result_text[:3000]}\n```", parse_mode="Markdown")

    except Exception as e:
        await target.reply_text(f"❌ Error processing summary for `{folder_name}`: {e}")

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
    
    query = update.callback_query
    if query:
        await query.answer()
        folder_name = query.data.replace("start_", "")
    else:
        if not context.args:
            await update.message.reply_text("Usage: `/start [number]`")
            return
        
        matches = find_all_folders_by_number(context.args[0])
        
        if len(matches) == 0:
            await update.message.reply_text(f"❌ No folders found starting with `{context.args[0]}_`")
            return
        
        if len(matches) > 1:
            keyboard = [[InlineKeyboardButton(m, callback_data=f"start_{m}")] for m in matches]
            await update.message.reply_text(
                f"🚀 Found multiple options for `{context.args[0]}`. Which one should I start?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return
        
        folder_name = matches[0]

    # --- Trigger the Execution Logic ---
    await execute_experiment_in_screen(update, folder_name)

async def execute_experiment_in_screen(update, folder_name):
    target = update.callback_query.message if update.callback_query else update.message
    
    folder_full_path = os.path.join(STATE_DIR, folder_name)
    script_path = os.path.join(folder_full_path, "WIP_tests_Transformer.py")
    bot_path = os.path.abspath(__file__)

    if os.path.exists(script_path):
        # The bash command with the '||' (OR) alert hook
        inner_cmd = (
            f"cd {folder_full_path} && "
            f"python3 WIP_tests_Transformer.py || "
            f"python3 {bot_path} --alert {folder_name}"
        )
        
        subprocess.run(["screen", "-dmS", folder_name, "-h", "10000", "bash", "-c", inner_cmd])
        await target.reply_text(f"✅ Started `{folder_name}`.\nYou'll be notified if it fails.")
    else:
        await target.reply_text(f"❌ Script not found: `{script_path}`")

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
    # --- Emergency Alert CLI Mode ---
    if len(sys.argv) > 2 and sys.argv[1] == "--alert":
        import asyncio
        from telegram import Bot
        async def send_emergency():
            bot = Bot(token="YOUR_TOKEN")
            await bot.send_message(chat_id=MY_USER_ID, text=f"🚨 **CRASH:** Experiment `{sys.argv[2]}` failed.")
        asyncio.run(send_emergency())
        sys.exit(0)

    # --- Standard Bot Mode ---
    app = Application.builder().token("YOUR_TOKEN").job_queue(None).build()
    
    # Handlers
    app.add_handler(CommandHandler("start", start_exp))
    app.add_handler(CommandHandler("summary", summary)) # Existing
    
    # Callback Handlers for Buttons
    app.add_handler(CallbackQueryHandler(start_exp, pattern="^start_"))
    app.add_handler(CallbackQueryHandler(summary, pattern="^sum_"))
    
    print("🚀 Bot is listening with selection and crash alerts...")
    app.run_polling()