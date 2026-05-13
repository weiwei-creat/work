#!/usr/bin/env python3
"""Flask server wrapping Microsoft TRELLIS for SAGE 3D object generation."""

import os, sys, io, json, time, random, argparse, threading, socket, subprocess

import torch, numpy as np, trimesh
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

sys.path.insert(0, os.environ.get('PYTHONPATH', '').split(':')[0])

from trellis.pipelines import TrellisTextTo3DPipeline

app = Flask(__name__)
CORS(app)

start_time = time.time()
pipeline = None
worker_jobs = {}
worker_jobs_lock = threading.Lock()


def get_network_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]; s.close()
        return ip
    except:
        try: return subprocess.check_output(['hostname', '-I'], text=True).strip().split()[0]
        except: return '0.0.0.0'


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'ok',
        'service': 'SAGE TRELLIS server',
        'health': '/health',
        'generate': '/generate',
    })


@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return ('', 204)


def mesh_to_glb_bytes(mesh_obj):
    """Convert MeshExtractResult to GLB bytes using trimesh directly (no nvdiffrast)."""
    if hasattr(mesh_obj, 'vertices') and hasattr(mesh_obj, 'faces'):
        verts = mesh_obj.vertices.cpu().numpy()
        faces = mesh_obj.faces.cpu().numpy()
        tm = trimesh.Trimesh(vertices=verts, faces=faces)
        return trimesh.exchange.gltf.export_glb(tm)
    return None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'gpu_available': torch.cuda.is_available(), 'pipeline_loaded': pipeline is not None})


@app.route('/api/v1/models', methods=['GET'])
def list_models():
    return jsonify({'object': 'list', 'data': [{'id': 'microsoft/TRELLIS-text-large', 'object': 'model', 'created': int(start_time)}]})


@app.route('/generate', methods=['POST'])
def generate_3d_model():
    try:
        data = request.get_json()
        input_text = data.get('input_text', data.get('prompt', 'A simple 3D object'))
        seed = data.get('seed', random.randint(1, 1000000))
        job_id = f'{int(time.time()*1000)}_{seed}'

        with worker_jobs_lock:
            worker_jobs[job_id] = {'status': 'processing', 'input_text': input_text, 'seed': seed, 'created_at': time.time()}

        print(f'Job {job_id}: "{input_text}" (seed={seed})')

        def generate_async():
            t0 = time.time()
            try:
                outputs = pipeline.run(prompt=input_text, seed=seed, formats=['gaussian', 'mesh'])

                glb_bytes = None
                mesh_list = outputs.get('mesh', [])
                if mesh_list and len(mesh_list) > 0:
                    glb_bytes = mesh_to_glb_bytes(mesh_list[0])

                elapsed = time.time() - t0
                size_mb = len(glb_bytes) / 1e6 if glb_bytes else 0
                print(f'Job {job_id}: Done in {elapsed:.1f}s, {size_mb:.1f}MB')

                with worker_jobs_lock:
                    worker_jobs[job_id] = {'status': 'completed', 'file_content': glb_bytes, 'generation_time': elapsed}

            except Exception as e:
                import traceback; traceback.print_exc()
                with worker_jobs_lock:
                    worker_jobs[job_id] = {'status': 'failed', 'error': str(e)}

        threading.Thread(target=generate_async, daemon=True).start()
        return jsonify({'status': 'accepted', 'job_id': job_id, 'message': 'Processing started'}), 202

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/job/<job_id>', methods=['GET'])
def get_job(job_id):
    with worker_jobs_lock:
        job = worker_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job['status'] == 'completed':
        if job['file_content'] is None:
            return jsonify({'error': 'Empty output'}), 500
        return send_file(io.BytesIO(job['file_content']), mimetype='application/octet-stream', as_attachment=True, download_name=f'model_{job_id}.glb')
    elif job['status'] == 'failed':
        return jsonify({'status': 'failed', 'error': job.get('error')}), 500
    return jsonify({'status': job['status'], 'job_id': job_id}), 202


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--pipeline-path', type=str, default='/home/gaok/coding/TRELLIS')
    args = parser.parse_args()

    print(f'Loading TRELLIS pipeline from {args.pipeline_path}...')
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.pipeline_path)
    pipeline.cuda()
    print('Pipeline loaded!')

    ip = get_network_ip()
    print(f'Server: http://{ip}:{args.port}')
    app.run(host='0.0.0.0', port=args.port, debug=False)
