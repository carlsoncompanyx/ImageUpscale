import os
import io
import uuid
import base64
import cv2
import traceback
import runpod
import requests

from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
from schemas.input import INPUT_SCHEMA

# Use /workspace for the Docker volume mount path
worker_build_env = os.getenv('WORKER_BUILD_ENV')
if worker_build_env:
    VOLUME_PATH = '/workspace'
else:
    VOLUME_PATH = os.getcwd()

GPU_ID = 0
TMP_PATH = f'{VOLUME_PATH}/tmp'
MODELS_PATH = f'{VOLUME_PATH}/models/ESRGAN'
GFPGAN_MODEL_PATH = f'{VOLUME_PATH}/models/GFPGAN/GFPGANv1.3.pth'
logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def upscale(
    source_image_path,
    image_extension,
    model_name='RealESRGAN_x4plus',
    outscale=4,
    face_enhance=False,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    denoise_strength=0.5
):
    """
    model_name options:
        - RealESRGAN_x4plus
        - RealESRNet_x4plus
        - RealESRGAN_x4plus_anime_6B
        - RealESRGAN_x2plus
        - 4x-UltraSharp
        - lollypop

    image_extension: .jpg or .png
    outscale: The final upsampling scale of the image
    """

    model_name = model_name.split('.')[0]

    if image_extension == '.jpg':
        image_format = 'JPEG'
    elif image_extension == '.png':
        image_format = 'PNG'
    else:
        raise ValueError('Unsupported image type, must be either JPEG or PNG')

    if model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus', '4x-UltraSharp', 'lollypop']:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    model_path = os.path.join(MODELS_PATH, model_name + '.pth')

    if not os.path.isfile(model_path):
        raise RuntimeError(f'Could not find model: {model_path}')

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        gpu_id=GPU_ID
    )

    face_enhancer = None
    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    img = cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f'Source image ({source_image_path}) is corrupt')

    try:
        if face_enhance and face_enhancer is not None:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        raise RuntimeError(e)
    else:
        result_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format=image_format)
        image_data = output_buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')


def determine_file_extension_from_b64(image_b64: str) -> str:
    """
    Very simple heuristic based on base64 header.
    Defaults to PNG if unsure.
    """
    try:
        if image_b64.startswith('/9j/'):
            return '.jpg'
        if image_b64.startswith('iVBORw0KG'):
            return '.png'
    except Exception:
        pass
    return '.png'


def upscaling_api(input):
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH, exist_ok=True)

    unique_id = uuid.uuid4()
    source_image_data = input['source_image']
    model_name = input['model']
    outscale = input['scale']
    face_enhance = input['face_enhance']
    tile = input['tile']
    tile_pad = input['tile_pad']
    pre_pad = input['pre_pad']
    half = input['half']

    source_file_extension = determine_file_extension_from_b64(source_image_data)
    source_bytes = base64.b64decode(source_image_data)
    source_image_path = f'{TMP_PATH}/source_{unique_id}{source_file_extension}'

    # Save original to disk
    with open(source_image_path, 'wb') as f:
        f.write(source_bytes)

    try:
        result_image_b64 = upscale(
            source_image_path,
            source_file_extension,
            model_name,
            outscale,
            face_enhance,
            tile,
            tile_pad,
            pre_pad,
            half
        )
    except Exception as e:
        logger.error(f'Upscale exception: {e}')
        logger.error(traceback.format_exc())

        # Clean up
        if os.path.exists(source_image_path):
            os.remove(source_image_path)

        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }

    # Clean up temp source
    if os.path.exists(source_image_path):
        os.remove(source_image_path)

    # ------------------------------------------------------------------
    # Upload result to Printify (CORRECT FORMAT)
    # ------------------------------------------------------------------
    printify_token = os.getenv('PRINTIFY_TOKEN')

    if not printify_token:
        return {
            "output": {
                "status": "missing_printify_token",
                "image_base64": result_image_b64
            }
        }

    try:
        headers = {
            "Authorization": f"Bearer {printify_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "file_name": f"upscaled_{unique_id}.jpg",
            "contents": result_image_b64   # <-- BASE64 STRING, Printify requires this
        }

        pr = requests.post(
            "https://api.printify.com/v1/uploads/images.json",
            headers=headers,
            json=payload,
            timeout=60
        )

        if pr.status_code != 200:
            logger.error(f"Printify upload failed {pr.status_code}: {pr.text}")
            return {
                "output": {
                    "status": "printify_error",
                    "status_code": pr.status_code,
                    "message": pr.text,
                    "image_base64": result_image_b64
                }
            }

        url = pr.json().get("src")

        return {
            "output": {
                "status": "ok",
                "printify_url": url,
                "image_base64": result_image_b64
            }
        }

    except Exception as e:
        logger.error(f"Printify exception: {e}")
        logger.error(traceback.format_exc())
        return {
            "output": {
                "status": "printify_exception",
                "message": str(e),
                "image_base64": result_image_b64
            }
        }



# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
def handler(event):
    validated_input = validate(event['input'], INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {
            'errors': validated_input['errors']
        }

    return upscaling_api(validated_input['validated_input'])


if __name__ == "__main__":
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start({"handler": handler})
