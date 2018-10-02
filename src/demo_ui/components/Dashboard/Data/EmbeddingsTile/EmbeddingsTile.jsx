import React from 'react';
import PropTypes from 'prop-types';

import './embeddings-tile.scss';

const EmbeddingTile = ({ name, dimensionality, vocabSize, corpus, tokens, plotSrc }) =>
  (
    <div className="embedding-tile tile">
      <div className="tile-header">
        <h3>{name} - {corpus}</h3>
      </div>
      <div className="tile-body">
        <div className="embedding-stats">
          <h4>Tokens</h4>
          <p>{tokens}</p>
          <h4>Vocab Size</h4>
          <p>{vocabSize}</p>
          <h4>Dimensionality Options</h4>
          <p>{dimensionality.join(', ')}</p>
        </div>
        <div className="embedding-plot">
          <h4>TSNE Plot (Top 100 Words)</h4>
          <a href={`http://localhost:8081/${plotSrc}`} target="_blank">
            <img src={plotSrc} width={200} height={200} alt="TSNE-Plot" />
          </a>
        </div>
      </div>
    </div>
  );

EmbeddingTile.propTypes = {
  name: PropTypes.string.isRequired,
  corpus: PropTypes.string.isRequired,
  dimensionality: PropTypes.arrayOf(PropTypes.number).isRequired,
  vocabSize: PropTypes.number.isRequired,
  tokens: PropTypes.string.isRequired,
  plotSrc: PropTypes.string.isRequired,
};

export default EmbeddingTile;
