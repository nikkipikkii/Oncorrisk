import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import Icon from './AppIcon';
import Button from './ui/Button';

const Header = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  const navigationItems = [
    { path: '/interactive-landing-hub', label: 'Home' },
    { path: '/riskprofile', label: 'Patient Risk Profile Inference' },
    { path: '/clinical-implementation', label: 'Gene Narrative' },
    { path: '/development-performance', label: 'Model Performance' },
    { path: '/analytics', label: 'Analytics' },
  ];

  const isActivePath = (path) => location?.pathname === path;

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  return (
    <header className="header-container">
      <div className="header-content">
        <nav className="header-nav">
          <Link to="/interactive-landing-hub" className="header-logo-link">
            <div className="header-logo-icon">
              <Icon name="Dna" size={24} color="var(--color-accent)" />
            </div>
            <span className="header-logo-text">OncoRisk</span>
          </Link>

          <div className="header-nav-desktop">
            {navigationItems?.map((item) => (
              <Link
                key={item?.path}
                to={item?.path}
                className={`header-nav-link ${
                  isActivePath(item?.path) ? 'header-nav-link-active' : ''
                }`}
              >
                {item?.label}
              </Link>
            ))}
          </div>

          <div className="header-cta-container">
            {/* Link to Gene Database */}
            <Link to="/clinical-implementation">
              <Button
                variant="outline"
                size="sm"
                iconName="Database"
                iconPosition="left"
              >
                Gene Database
              </Button>
            </Link>

            {/* Link to Test Model */}
            <Link to="/riskprofile">
              <Button
                variant="default"
                size="sm"
                iconName="FlaskConical"
                iconPosition="left"
              >
                Test Model
              </Button>
            </Link>
          </div>

          <button
            onClick={toggleMobileMenu}
            className="header-mobile-toggle"
            aria-label="Toggle mobile menu"
            aria-expanded={isMobileMenuOpen}
          >
            <Icon
              name={isMobileMenuOpen ? 'X' : 'Menu'}
              size={24}
              color="var(--color-foreground)"
            />
          </button>
        </nav>

        {isMobileMenuOpen && (
          <div className="header-mobile-menu animate-fadeIn">
            <div className="header-mobile-nav">
              {navigationItems?.map((item) => (
                <Link
                  key={item?.path}
                  to={item?.path}
                  onClick={closeMobileMenu}
                  className={`header-mobile-link ${
                    isActivePath(item?.path) ? 'header-mobile-link-active' : ''
                  }`}
                >
                  {item?.label}
                </Link>
              ))}
              <div className="header-mobile-cta">
                <Link to="/clinical-implementation" onClick={closeMobileMenu}>
                  <Button
                    variant="outline"
                    fullWidth
                    iconName="Database"
                    iconPosition="left"
                  >
                    Gene Database
                  </Button>
                </Link>
                <Link to="/riskprofile" onClick={closeMobileMenu}>
                  <Button
                    variant="default"
                    fullWidth
                    iconName="FlaskConical"
                    iconPosition="left"
                  >
                    Test Model
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;