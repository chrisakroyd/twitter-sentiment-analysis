import React from 'react';
import PropTypes from 'prop-types';
import shortId from 'shortid';
import './data.scss';

import DatasetTile from './DatasetTile/DatasetTile';
import EmbeddingsTile from './EmbeddingsTile/EmbeddingsTile';

import datasetShape from '../../../prop-shapes/datasetShape';
import embeddingsShape from '../../../prop-shapes/embeddingsShape';

class Data extends React.Component {
  createDatasetTiles() {
    return this.props.datasets.map(data => (
      <DatasetTile
        key={shortId.generate()}
        name={data.name}
        description={data.description}
        size={data.size}
        words={data.words}
        statistics={data.statistics}
      />
    ));
  }

  createEmbeddingsTiles() {
    return this.props.embeddings.map(data => (
      <EmbeddingsTile
        key={shortId.generate()}
        name={data.name}
        corpus={data.corpus}
        dimensionality={data.dimensionality}
        vocabSize={data.vocabSize}
        tokens={data.tokens}
        plotSrc={data.plotSrc}
      />
    ));
  }

  render() {
    return (
      <div className="data dash-body">
        <div className="body-header">
          <h1>Data</h1>
        </div>
        <div className="tile-container">
          <h2 className="data-header">Datasets</h2>
          <div className="tile-row">
            {this.createDatasetTiles()}
          </div>

          <h2 className="data-header">Embeddings</h2>
          <div className="tile-row">
            {this.createEmbeddingsTiles()}
          </div>
        </div>
      </div>
    );
  }
}

Data.propTypes = {
  datasets: PropTypes.arrayOf(datasetShape).isRequired,
  embeddings: PropTypes.arrayOf(embeddingsShape).isRequired,
};

export default Data;
