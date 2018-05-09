import React from 'react';
import PropTypes from 'prop-types';
import { Route } from 'react-router';

import Sidebar from '../../components/Dashboard/Sidebar/Sidebar';

import Demo from './Demo/Demo';
import Data from '../../components/Dashboard/Data/Data';
import Models from '../../components/Dashboard/Models/Models';
import Train from '../../components/Dashboard/Train/Train';
import Results from '../../components/Dashboard/Results/Results';

import genSidebarDescriptors from './genSidebarDescriptors';

import './dashboard.scss';

class Dashboard extends React.Component {
  render() {
    return (
      <div className="dashboard">
        <Sidebar
          highlightedItem={this.props.activeView}
          links={genSidebarDescriptors(this.props.match)}
          onSidebarLink={this.props.onDashClick}
        />
        <div>
          <Route
            exact
            path="/"
            render={() => (
              <Demo
                status={this.props.status}
                activeText={this.props.activeText}
                process={this.props.process}
                setText={this.props.setText}
                tweets={this.props.tweets}
                loadNewTweets={this.props.tweetRefresh}
              />
            )}
          />

          <Route
            exact
            path="/data"
            render={() => (
              <Data
                datasets={this.props.datasets.datasets}
                embeddings={this.props.embeddings.embeddings}
              />
            )}
          />

          <Route
            exact
            path="/models"
            render={() => (
              <Models models={this.props.models.models} />
            )}
          />

          <Route
            exact
            path="/train"
            render={() => (
              <Train />
            )}
          />

          <Route
            exact
            path="/results"
            render={() => (
              <Results results={this.props.results.results} />
            )}
          />
        </div>
      </div>
    );
  }
}

Dashboard.propTypes = {
  // Functions
  onDashClick: PropTypes.func.isRequired,
  process: PropTypes.func.isRequired,
  setText: PropTypes.func.isRequired,
  tweetRefresh: PropTypes.func.isRequired,
  // Data
  activeView: PropTypes.string.isRequired,
  activeText: PropTypes.shape({}).isRequired,
  embeddings: PropTypes.shape({}).isRequired,
  datasets: PropTypes.shape({}).isRequired,
  models: PropTypes.shape({}).isRequired,
  results: PropTypes.shape({}).isRequired,
  tweets: PropTypes.shape({}).isRequired,
  status: PropTypes.shape({}).isRequired,
};


export default Dashboard;
