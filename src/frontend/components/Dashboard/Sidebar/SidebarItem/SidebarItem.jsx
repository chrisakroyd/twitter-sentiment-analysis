import React from 'react';
import ClassNames from 'classnames';
import PropTypes from 'prop-types';

import './sidebar-item.scss';

class SidebarLink extends React.Component {
  onClick() {
    this.props.onSidebarLinkClick(this.props.label, this.props.type, this.props.link);
  }

  render() {
    const iconClass = ClassNames('material-icons', { highlighted: this.props.isHighlighted });
    const itemLabelClass = ClassNames('item-label', { highlighted: this.props.isHighlighted });

    return (
      <div className="sidebar-item" onClick={() => this.onClick()}>
        <i className={iconClass}>{this.props.icon}</i>
        <p className={itemLabelClass}>{this.props.label}</p>
      </div>
    );
  }
}

SidebarLink.propTypes = {
  icon: PropTypes.string.isRequired,
  label: PropTypes.string.isRequired,
  link: PropTypes.string.isRequired,
  isHighlighted: PropTypes.bool.isRequired,
  type: PropTypes.string.isRequired,
  onSidebarLinkClick: PropTypes.func.isRequired,
};

export default SidebarLink;
