import React from 'react';
import PropTypes from 'prop-types';
import shortid from 'shortid';

import SidebarLink from './SidebarItem/SidebarItem';
import linkDescriptorShape from '../../../prop-shapes/linkDescriptorShape';

import './sidebar.scss';

class Sidebar extends React.Component {
  getLinks() {
    const links = [];
    this.props.links.forEach((link, i) => {
      links.push((
        <SidebarLink
          key={shortid.generate()}
          icon={link.icon}
          label={link.label}
          link={link.linkTo}
          isHighlighted={link.type === this.props.highlightedItem}
          type={link.type}
          onSidebarLinkClick={this.props.onSidebarLink}
        />
      ));
    });

    return links;
  }

  render() {
    return (
      <nav className="sidebar sidebar-open">
        <div>
          {this.getLinks()}
        </div>
      </nav>
    );
  }
}

Sidebar.propTypes = {
  links: PropTypes.arrayOf(linkDescriptorShape).isRequired,
  highlightedItem: PropTypes.string.isRequired,
  onSidebarLink: PropTypes.func.isRequired,
};

export default Sidebar;
