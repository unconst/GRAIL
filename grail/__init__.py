#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import os
import sys
import json
import time
import click
import random
import asyncio
import logging
import hashlib
import traceback
import bittensor as bt
from dotenv import load_dotenv
from botocore.config import Config
from collections import defaultdict
from abc import ABC, abstractmethod
from aiobotocore.session import get_session
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__version__ = "0.0.0"

from .grail import Prover, Verifier

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
NETUID = 120
WINDOW_LENGTH = 5  # Generate inferences every 5 blocks
TRACE  = 5
logging.addLevelName(TRACE, "TRACE")

# Model configuration
LLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Using TinyLlama 1B model

# --------------------------------------------------------------------------- #
#                               Logging                                       #
# --------------------------------------------------------------------------- #
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("grail")
def setup_logging(verbosity: int):
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    
    # GRAIL debug details only visible with -vv or higher
    if verbosity < 2:
        logging.getLogger("grail").setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
def get_conf(key, default=None) -> Any:
    v = os.getenv(key); 
    if not v and default is None:
        click.echo(f"{key} not set.\nRun:\n\taf set {key} <value>", err=True); sys.exit(1)
    return v or default

# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.trace("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor()
        await SUBTENSOR.initialize()
        logger.trace("Connected")
    return SUBTENSOR


# --------------------------------------------------------------------------- #
#                   S3 helpers                                                #
# --------------------------------------------------------------------------- #
get_client_ctx = lambda: get_session().create_client(
    "s3",
    endpoint_url=f"https://{get_conf('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=get_conf("R2_WRITE_ACCESS_KEY_ID"),
    aws_secret_access_key=get_conf("R2_WRITE_SECRET_ACCESS_KEY"),
    config=Config(max_pool_connections=256)
)

async def sink_window_inferences(wallet: bt.wallet, window_start: int, inferences: List[dict]):
    """Upload window of inferences to S3"""
    key = f"grail/windows/{wallet.hotkey.ss58_address}-window-{window_start}.json"
    
    # Pack all inferences into window data
    window_data = {
        "wallet": wallet.hotkey.ss58_address,
        "window_start": window_start,
        "window_length": WINDOW_LENGTH,
        "inference_count": len(inferences),
        "inferences": inferences,
        "timestamp": time.time()
    }
    
    body = json.dumps(window_data).encode()
    logger.debug(f"[SINK] window={window_start} count={len(inferences)} → key={key}")
    try:
        async with get_client_ctx() as client:
            await client.put_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key, Body=body)
    except Exception: 
        logger.error("R2 write failed for %s", key, exc_info=True)

async def file_exists(key: str) -> bool:
    """Check if a file exists in the bucket without downloading it"""
    try:
        async with get_client_ctx() as client:
            await client.head_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key)
            return True
    except Exception:
        return False

async def list_bucket_files(prefix: str) -> List[str]:
    """List files in bucket with given prefix"""
    try:
        async with get_client_ctx() as client:
            response = await client.list_objects_v2(
                Bucket=get_conf("R2_BUCKET_ID"), 
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
    except Exception:
        logger.error("Failed to list bucket files with prefix %s", prefix, exc_info=True)
        return []

async def get_file(key: str):
    """Download file from S3"""
    try:
        async with get_client_ctx() as client:
            resp = await client.get_object(Bucket=get_conf("R2_BUCKET_ID"), Key=key)
            data = await resp["Body"].read()
        return json.loads(data) if key.endswith('.json') else data
    except Exception:
        logger.error("Failed to get file %s", key, exc_info=True)
        return None

# --------------------------------------------------------------------------- #
#                        Helper Functions                                     #
# --------------------------------------------------------------------------- #
def generate_prompt(hotkey_address: str, block_hash: str, nonce: int) -> str:
    """Generate prompt in the required format"""
    return f"Hey my name is {hotkey_address} it is currently {block_hash} days since friday and my fav number is {nonce}, tell me a story about these three facts"

def parse_filename(filename: str) -> Tuple[str, int, int]:
    """Parse filename to extract wallet, block, nonce"""
    # Remove prefix and extension
    basename = filename.split('/')[-1].replace('.json', '')
    parts = basename.split('-')
    if len(parts) >= 3:
        wallet = parts[0]
        block = int(parts[1])
        nonce = int(parts[2])
        return wallet, block, nonce
    return None, None, None

def parse_window_filename(filename: str) -> Tuple[str, int]:
    """Parse window filename to extract wallet and window_start"""
    # Remove prefix and extension
    basename = filename.split('/')[-1].replace('.json', '')
    # Format: {wallet}-window-{window_start}
    parts = basename.split('-')
    if len(parts) >= 3 and parts[1] == 'window':
        wallet = parts[0]
        window_start = int(parts[2])
        return wallet, window_start
    return None, None

def sign_inference(inference_data: dict, wallet: bt.wallet) -> dict:
    """Sign an inference using the wallet hotkey"""
    # Create challenge string from key inference data
    challenge = f"{inference_data['prompt']}{inference_data['block_hash']}{inference_data['nonce']}"
    inference_data['challenge'] = challenge
    inference_data['hotkey'] = wallet.hotkey.ss58_address
    inference_data['signature'] = wallet.hotkey.sign(data=challenge).hex()
    return inference_data

def verify_inference_signature(inference_data: dict) -> bool:
    """Verify the signature of an inference"""
    try:
        challenge = inference_data.get('challenge')
        hotkey = inference_data.get('hotkey')
        signature = inference_data.get('signature')
        
        if not all([challenge, hotkey, signature]):
            return False
            
        keypair = bt.Keypair(ss58_address=hotkey)
        return keypair.verify(data=challenge, signature=bytes.fromhex(signature))
    except Exception:
        return False

def derive_secret_key(hotkey_address: str) -> bytes:
    """Derive deterministic secret key from hotkey for verification"""
    return hashlib.sha256(f"grail_secret_{hotkey_address}".encode()).digest()

# Global storage for miner state
miner_inference_counts = defaultdict(list)  # track inferences per block for weight calculation

# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """GRAIL CLI"""
    setup_logging(verbose)
    
# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #
HEARTBEAT = time.monotonic()
async def watchdog(timeout: int = 300):
    global HEARTBEAT
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)
            
# --------------------------------------------------------------------------- #
#                               MINER                                         #
# --------------------------------------------------------------------------- #
@cli.command("mine")
def mine():    
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize model and prover
    logger.info(f"Loading model: {LLAMA_MODEL}")
    prover = Prover(model_name=LLAMA_MODEL)
    # Set deterministic secret key based on hotkey
    prover.secret_key = derive_secret_key(wallet.hotkey.ss58_address)
    
    async def _run():
        subtensor = None
        last_window_start = -1
        
        while True:
            try:
                global HEARTBEAT; HEARTBEAT = time.monotonic()
                if subtensor is None: 
                    subtensor = await get_subtensor()                
                current_block = await subtensor.get_current_block()
                
                # Calculate current window start (blocks divisible by WINDOW_LENGTH)
                window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                
                # Only process if we're in a new window
                if window_start <= last_window_start:
                    await asyncio.sleep(2)  # Wait for new window
                    continue
                
                logger.info(f"🔥 Starting inference generation for window {window_start}-{window_start + WINDOW_LENGTH - 1}")
                window_block_hash = await subtensor.get_block_hash(window_start)
                
                # Generate as many inferences as possible during this window
                inferences = []
                start_time = time.time()
                inference_count = 0
                
                # Generate inferences until the window closes
                while True:
                    current_block = await subtensor.get_current_block()
                    current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                    
                    # Stop if we've moved to the next window
                    if current_window > window_start:
                        break
                    
                    try:
                        inference_count += 1
                        print(f"\r⚡ Generating inference {inference_count}...", end="", flush=True)
                        
                        # Generate random nonce
                        nonce = random.randint(1000, 9999)
                        
                        # Create prompt in required format
                        prompt = generate_prompt(wallet.hotkey.ss58_address, window_block_hash, nonce)
                        
                        # Generate inference and proof using GRAIL with window block hash as randomness
                        commit_data = prover.commit(prompt, window_block_hash, max_new_tokens=32)
                        proof_data = prover.open(window_block_hash)
                        
                        # Prepare inference data
                        inference_data = {
                            "window_start": window_start,
                            "block": current_block,
                            "nonce": nonce,
                            "prompt": prompt,
                            "block_hash": window_block_hash,
                            "commit": commit_data,
                            "proof": proof_data,
                            "timestamp": time.time()
                        }
                        
                        # Sign the inference
                        inference_data = sign_inference(inference_data, wallet)
                        inferences.append(inference_data)
                        
                        # Small delay to prevent overwhelming the system
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate inference {inference_count}: {e}")
                        continue
                
                # Clear the progress line
                print("\r" + " " * 50 + "\r", end="")
                
                elapsed_time = time.time() - start_time
                logger.info(f"🎯 Generated {len(inferences)} inferences in {elapsed_time:.1f}s for window {window_start}")
                
                if inferences:
                    # Upload all inferences as a single window file
                    await sink_window_inferences(wallet, window_start, inferences)
                    logger.info(f"📤 Uploaded window {window_start} with {len(inferences)} inferences")
                else:
                    logger.warning(f"No inferences generated for window {window_start}")
                
                last_window_start = window_start
                
            except asyncio.CancelledError: 
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in miner loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
                
    async def main():
        await asyncio.gather(
            _run(),
            watchdog()
        )
    asyncio.run(main())

# --------------------------------------------------------------------------- #
#                               Validator                                     #
# --------------------------------------------------------------------------- #
@cli.command("validate")
def validate():
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize verifier
    logger.info(f"Loading model for validation: {LLAMA_MODEL}")
    verifier = Verifier(model_name=LLAMA_MODEL)
    
    # Storage for inference counts per miner
    inference_counts = defaultdict(lambda: defaultdict(int))  # {hotkey: {window: count}}
    
    async def _run():
        subtensor = None
        last_processed_window = -1
        
        while True:
            try:
                global HEARTBEAT; HEARTBEAT = time.monotonic()
                if subtensor is None: 
                    subtensor = await get_subtensor()

                meta = await subtensor.metagraph(NETUID)
                current_block = await subtensor.get_current_block()
                
                # Calculate current and previous windows
                current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                # Process the previous complete window
                target_window = current_window - WINDOW_LENGTH
                
                if target_window <= last_processed_window or target_window < 0:
                    await asyncio.sleep(5)  # Wait for new window
                    continue
                
                logger.info(f"🔍 Processing window {target_window}-{target_window + WINDOW_LENGTH - 1}")
                
                # Get block hash for the window start
                target_window_hash = await subtensor.get_block_hash(target_window)
                
                # Check for files from active hotkeys only
                logger.info(f"Checking files for {len(meta.hotkeys)} active hotkeys in window {target_window}")
                
                # Download and process files only from registered hotkeys
                total_valid_inferences = 0
                window_inference_counts = defaultdict(int)
                files_found = 0
                
                for wallet_addr in meta.hotkeys:
                    try:
                        # Construct expected filename for this hotkey and window
                        filename = f"grail/windows/{wallet_addr}-window-{target_window}.json"
                        
                        # Check if file exists before downloading
                        if not await file_exists(filename):
                            continue
                        
                        files_found += 1
                        logger.debug(f"Found file for hotkey {wallet_addr}")
                        
                        window_data = await get_file(filename)
                        if not window_data:
                            logger.warning(f"Could not download {filename}")
                            continue
                        
                        file_wallet_addr = window_data.get("wallet")
                        window_start = window_data.get("window_start")
                        inferences = window_data.get("inferences", [])
                        
                        # Basic window validation
                        if file_wallet_addr != wallet_addr:
                            logger.warning(f"Wallet mismatch in {filename}: expected {wallet_addr}, got {file_wallet_addr}")
                            continue
                        
                        if window_start != target_window:
                            logger.warning(f"Window mismatch in {filename}: expected {target_window}, got {window_start}")
                            continue
                        
                        # Verify all inferences in the window
                        valid_count = 0
                        nonces_seen = set()
                        
                        for inference in inferences:
                            try:
                                # Check required fields
                                required_fields = ["window_start", "nonce", "prompt", "block_hash", "commit", "proof", "challenge", "hotkey", "signature"]
                                if not all(field in inference for field in required_fields):
                                    logger.debug(f"Missing required fields in inference from {wallet_addr}")
                                    continue
                                
                                # Check window consistency
                                if inference["window_start"] != target_window:
                                    logger.debug(f"Window mismatch in inference from {wallet_addr}")
                                    continue
                                
                                # Check block hash matches
                                if inference["block_hash"] != target_window_hash:
                                    logger.debug(f"Block hash mismatch in inference from {wallet_addr}")
                                    continue
                                
                                # Check nonce uniqueness within window
                                nonce = inference["nonce"]
                                if nonce in nonces_seen:
                                    logger.debug(f"Duplicate nonce {nonce} in window from {wallet_addr}")
                                    continue
                                nonces_seen.add(nonce)
                                
                                # Verify signature
                                if not verify_inference_signature(inference):
                                    logger.debug(f"Invalid signature for inference from {wallet_addr}")
                                    continue
                                
                                # Check prompt format
                                expected_prompt = generate_prompt(wallet_addr, target_window_hash, nonce)
                                if inference["prompt"] != expected_prompt:
                                    logger.debug(f"Invalid prompt format in inference from {wallet_addr}")
                                    continue
                                
                                # Verify GRAIL proof (spot checking for efficiency)
                                if random.random() < 0.2:  # 20% spot check
                                    try:
                                        logger.debug(f"Spot checking GRAIL proof from {wallet_addr}")
                                        prover_secret_key = derive_secret_key(wallet_addr)
                                        is_valid = verifier.verify(inference["commit"], inference["proof"], prover_secret_key)
                                        if not is_valid:
                                            logger.debug(f"GRAIL proof verification failed for {wallet_addr}")
                                            continue
                                    except Exception as e:
                                        logger.debug(f"Proof verification error for {wallet_addr}: {e}")
                                        continue
                                
                                valid_count += 1
                                
                            except Exception as e:
                                logger.debug(f"Error processing inference from {wallet_addr}: {e}")
                                continue
                        
                        window_inference_counts[wallet_addr] = valid_count
                        total_valid_inferences += valid_count
                        
                        logger.info(f"✅ {wallet_addr}: {valid_count}/{len(inferences)} valid inferences")
                        
                    except Exception as e:
                        logger.warning(f"Error processing window file {filename}: {e}")
                        continue
                
                logger.info(f"📁 Found {files_found} window files from {len(meta.hotkeys)} active hotkeys")
                logger.info(f"🏁 Total valid inferences in window {target_window}: {total_valid_inferences}")
                
                # Update global inference counts for weight calculation
                for hotkey, count in window_inference_counts.items():
                    inference_counts[hotkey][target_window] = count
                
                # Compute weights based on moving average of inferences per window
                weights = []
                for uid, hotkey in enumerate(meta.hotkeys):
                    # Calculate moving average over last 3 windows
                    recent_windows = range(max(0, target_window - 2*WINDOW_LENGTH), target_window + 1, WINDOW_LENGTH)
                    total_inferences = sum(inference_counts[hotkey].get(w, 0) for w in recent_windows)
                    avg_inferences = total_inferences / len(recent_windows)
                    
                    # Weight based on inference count (normalize to 0-1)
                    # Assume max 50 inferences per window as reasonable target
                    weight = min(1.0, avg_inferences / 50.0)
                    weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [0.0] * len(meta.hotkeys)
                
                # Log non-zero weights
                non_zero_weights = [(meta.hotkeys[i], weights[i]) for i in range(len(weights)) if weights[i] > 0]
                if non_zero_weights:
                    logger.info(f"⚖️  Setting weights for {len(non_zero_weights)} miners")
                    for hotkey, weight in non_zero_weights[:5]:  # Show top 5
                        logger.info(f"   {hotkey}: {weight:.4f}")
                else:
                    logger.info("⚖️  No miners received weights this window")
                
                # Set weights on network
                await subtensor.set_weights(
                    wallet=wallet,
                    netuid=NETUID,
                    uids=meta.uids,
                    weights=weights,
                    wait_for_inclusion=False
                )
                
                last_processed_window = target_window
                
            except asyncio.CancelledError: 
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in validator loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
            
    async def main():
        await asyncio.gather(
            _run(),
            watchdog(timeout = (60 * 10))
        )
    asyncio.run(main())