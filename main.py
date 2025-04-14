# main.py (Modified)
import comfy.options
comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time
from comfy.cli_args import args
from app.logger import setup_logger
import itertools
import utils.extra_config
import logging
import sys
import threading # <-- Import threading
import uvicorn   # <-- Import uvicorn

# --- FastAPI App Import ---
try:
    # Assuming your FastAPI app instance is named 'app' in this file
    from api_server.fastapi.server import app as fastapi_app
except ImportError as e:
    print(f"Error importing FastAPI app: {e}")
    print("Please ensure api_server/fastapi/server.py exists and contains an 'app' instance.")
    sys.exit(1)
# --- End FastAPI App Import ---

# --- Original ComfyUI Imports and Setup ---
import cuda_malloc
import comfy.utils
import execution
import server
from server import BinaryEventTypes
import nodes
import comfy.model_management
import comfyui_version
import app.logger
# --- End Original ComfyUI Imports and Setup ---


# --- Environment Setup ---
# NOTE: These do not do anything on core ComfyUI which should already have no communication with the internet, they are for custom nodes.
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)
# --- End Environment Setup ---


# --- FastAPI Server Function ---
def run_fastapi_server():
    """
    Runs the FastAPI Uvicorn server.
    """
    print("Starting FastAPI server via Uvicorn on port 8000...")
    try:
        config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        # Uvicorn's server.run() is blocking and needs to be run in the event loop
        # managed by the thread. We'll run it directly here.
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        logging.error(f"FastAPI server failed: {e}")
        logging.error(traceback.format_exc())
# --- End FastAPI Server Function ---


# --- ComfyUI Helper Functions (Keep original functions) ---
def apply_custom_paths():
    # (Keep the original function content)
    # extra model paths
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    # --output-directory, --input-directory, --user-directory
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)


def execute_prestartup_script():
    # (Keep the original function content)
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    if args.disable_all_custom_nodes:
        return

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        logging.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

apply_custom_paths()
execute_prestartup_script()

# --- Main ComfyUI Code (Keep original functions) ---
import asyncio
import shutil
import gc

if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

# (Keep cuda_malloc_warning, prompt_worker, run, hijack_progress, cleanup_temp functions as they are)
def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


def prompt_worker(q, server_instance):
    current_time: float = 0.0
    e = execution.PromptExecutor(server_instance, lru_size=args.cache_lru)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id

            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            q.task_done(item_id,
                        e.history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))
            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose), server_instance.publish_loop()
    )


def hijack_progress(server_instance):
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()
        progress = {"value": value, "max": total, "prompt_id": server_instance.last_prompt_id, "node": server_instance.last_node_id}

        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            server_instance.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server_instance.client_id)

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- Modified start_comfyui to run in a thread ---
def run_comfyui_server():
    """
    Initializes and starts the main ComfyUI server components.
    This function now contains the logic previously in start_comfyui and the main block.
    """
    logging.info("Starting ComfyUI server...")
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    # Get or create event loop for this thread
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    prompt_server = server.PromptServer(loop)
    q = execution.PromptQueue(prompt_server)

    nodes.init_extra_nodes(init_custom_nodes=not args.disable_all_custom_nodes)

    cuda_malloc_warning()

    prompt_server.add_routes()
    hijack_progress(prompt_server)

    threading.Thread(target=prompt_worker, daemon=True, args=(q, prompt_server,)).start()

    if args.quick_test_for_ci:
        logging.info("Quick test mode enabled. Exiting.")
        # In a real scenario, might need a cleaner way to signal exit if needed.
        return # Exit this thread if quick test

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    async def start_all():
        await prompt_server.setup()
        await run(prompt_server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start)

    try:
        logging.info("Starting ComfyUI server in thread...")
        app.logger.print_startup_warnings()
        loop.run_until_complete(start_all())
    except KeyboardInterrupt:
        logging.info("\nComfyUI server thread stopped.")
    finally:
        cleanup_temp()
        loop.close()
        logging.info("ComfyUI server thread finished.")

# --- Main Execution Block ---
if __name__ == "__main__":
    logging.info("ComfyUI version: {}".format(comfyui_version.__version__))

    # --- Device and Deterministic Setup (Keep original) ---
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
        logging.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    if args.windows_standalone_build:
        try:
            from fix_torch import fix_pytorch_libomp
            fix_pytorch_libomp()
        except:
            pass
    # --- End Device and Deterministic Setup ---


    # --- Start Servers in Threads ---
    print("Main process starting...")

    # 1. Create thread for ComfyUI server
    comfyui_server_thread = threading.Thread(target=run_comfyui_server, daemon=True)

    # 2. Create thread for FastAPI server
    fastapi_server_thread = threading.Thread(target=run_fastapi_server, daemon=True)

    # 3. Start threads
    print("Starting ComfyUI server thread...")
    comfyui_server_thread.start()

    print("Starting FastAPI server thread...")
    fastapi_server_thread.start()

    # 4. Keep main thread alive
    try:
        while True:
            time.sleep(1)
            if not comfyui_server_thread.is_alive():
                logging.warning("ComfyUI server thread seems to have stopped unexpectedly.")
                break
            if not fastapi_server_thread.is_alive():
                logging.warning("FastAPI server thread seems to have stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nCtrl+C received in main thread. Shutting down...")
        # Threads are daemons, they will exit when the main thread exits.
        # Add any specific cleanup needed for the main thread if necessary.

    print("Main process finished.")
    # Ensure temp cleanup happens on exit if threads didn't handle it
    cleanup_temp()
