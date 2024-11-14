from flask import Flask, render_template, jsonify, request
import json
import os
import numpy as np
import threading
from train import train_model  # We'll create this function

app = Flask(__name__)

# Store training states for both models
model_states = {
    'model1': {
        'status': 'Not Started',
        'progress': None,
        'thread': None,
        'is_training': False
    },
    'model2': {
        'status': 'Not Started',
        'progress': None,
        'thread': None,
        'is_training': False
    }
}

@app.route('/')
def index():
    # Check if any model is currently training
    for model_id in model_states:
        if model_states[model_id]['thread'] is not None and model_states[model_id]['thread'].is_alive():
            model_states[model_id]['is_training'] = True
        else:
            model_states[model_id]['is_training'] = False
    return render_template('index.html', model_states=model_states)

@app.route('/start_training', methods=['POST'])
def start_training():
    config = request.json
    model_id = config['model_id']
    
    if model_states[model_id]['is_training']:
        return jsonify({'status': 'error', 'message': f'{model_id} is already training'})
    
    # Clear old results
    try:
        os.remove(f'static/{model_id}_progress.json')
        os.remove(f'static/{model_id}_samples.npy')
        os.remove(f'static/{model_id}_model.pth')
    except FileNotFoundError:
        pass
    
    # Start training in a separate thread
    thread = threading.Thread(
        target=train_model,
        args=(model_id, config)
    )
    thread.start()
    
    model_states[model_id]['thread'] = thread
    model_states[model_id]['status'] = 'Training'
    model_states[model_id]['is_training'] = True
    
    return jsonify({'status': 'success'})

@app.route('/progress/<model_id>')
def progress(model_id):
    try:
        with open(f'static/{model_id}_progress.json', 'r') as f:
            progress = json.load(f)
        return jsonify(progress)
    except FileNotFoundError:
        return jsonify({
            'training_losses': [],
            'validation_losses': [],
            'training_accuracy': [],
            'test_accuracy': [],
            'current_epoch': 0,
            'current_batch': 0,
            'status': 'Not Started'
        })

@app.route('/results/<model_id>')
def results(model_id):
    try:
        samples = np.load(f'static/{model_id}_samples.npy', allow_pickle=True)
        return jsonify({
            'samples': [{
                'true': int(s['true']),
                'pred': int(s['pred']),
                'true_label': s['true_label'],
                'pred_label': s['pred_label'],
                'image': s['image'].tolist()
            } for s in samples]
        })
    except FileNotFoundError:
        return jsonify({'samples': []})

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True) 