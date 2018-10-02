import React from 'react';

import Architecture from './Architecture/Architecture';
import './model-tile.scss';

import modelShape from '../../../../prop-shapes/modelShape';

const ModelTile = ({ model }) =>
  (
    <div className="model-tile tile">
      <div className="tile-header">
        <h3>{model.name} - {model.lastModified}</h3>
      </div>
      <div className="tile-body">
        <div className="architecture-container">
          <h3>Neural Visualisation</h3>
          <div className="visualisation">
            <Architecture architecture={model.architecture} />
          </div>
        </div>
        <div className="model-stats">
          <h3>Network Information</h3>
          <h4>Embeddings</h4>
          <p>glove.twitter.27b.200d.txt</p>
          <h4>Tokenizer</h4>
          <p>TokenizerAnnotateV2</p>
          <h4>Parameters</h4>
          <p>{model.size}</p>
          <h4>Time Per Epoch (Seconds)</h4>
          <p>{model.timePerEpoch}</p>
          <h4>Optimizer</h4>
          <p>{model.optimizer}</p>
          <h4>Learn Rate / Clip Norm / Noise / Dropout</h4>
          <p>{model.learnRate} / {model.clipNorm} / {model.noise} / {model.embedDrop}</p>
        </div>
      </div>
    </div>
  );

ModelTile.propTypes = {
  model: modelShape.isRequired,
};

export default ModelTile;
