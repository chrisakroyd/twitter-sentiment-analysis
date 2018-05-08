import React from 'react';

import './result-tile.scss';
import resultShape from '../../../../prop-shapes/resultShape';

const imageHeight = 225;
const imageWidth = 270;

const ResultTile = ({ result }) =>
  (
    <div className="result-tile tile">
      <div className="tile-header">
        <h3>{result.name} - {result.lastRun}</h3>
      </div>
      <div className="tile-body">
        <div className="confusion-matrixes">
          <h4>Train Confusion Matrix</h4>
          <a href={`http://localhost:8081/${result.trainConfMatrixSrc}`} target="_blank">
            <img
              src={result.trainConfMatrixSrc}
              width={imageWidth}
              height={imageHeight}
              alt="Train Confusion Matrix"
            />
          </a>
          <h4>Validation Confusion Matrix</h4>
          <a href={`http://localhost:8081/${result.valConfMatrixSrc}`} target="_blank">
            <img
              src={result.valConfMatrixSrc}
              width={imageWidth}
              height={imageHeight}
              alt="Validation Confusion Matrix"
            />
          </a>
        </div>
        <div className="result-stats">
          <h4>Last Modified</h4>
          <p>{result.lastModified}</p>
          <h4>Last Ran</h4>
          <p>{result.lastRun}</p>
          <h4>Runs</h4>
          <p>{result.runs}</p>
          <h4>Tokenization Scheme</h4>
          <p>{result.tokenizationScheme}</p>
          <h4>Train Metrics (Acc/F1/Loss)</h4>
          <p>{result.trainAcc}% / {result.trainF1}% / {result.trainLoss}</p>
          <h4>Validation Metrics (Acc/F1/Loss)</h4>
          <p>{result.valAcc}% / {result.valF1}% / {result.valLoss}</p>

          <h4>Train Improvement (Acc/F1/Loss)</h4>
          <p>
            <i className="down-arrow negative material-icons">play_arrow</i>{result.trainAccChange}% /
            <i className="down-arrow negative material-icons">play_arrow</i>{result.trainF1Change}% /
            <i className="down-arrow negative material-icons">play_arrow</i>{result.trainLossChange}
          </p>

          <h4>Validation Improvement (Acc/F1/Loss)</h4>
          <p>
            <i className="up-arrow positive material-icons">play_arrow</i>{result.valAccChange}% /
            <i className="up-arrow positive material-icons">play_arrow</i>{result.valF1Change}% /
            <i className="up-arrow positive material-icons">play_arrow</i>{result.valLossChange}
          </p>
        </div>
      </div>
    </div>
  );

ResultTile.propTypes = {
  result: resultShape.isRequired,
};

export default ResultTile;
