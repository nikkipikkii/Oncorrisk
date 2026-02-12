import React from 'react';
import { Link } from 'react-router-dom';
import Icon from './AppIcon';

const Footer = () => {
  const currentYear = new Date()?.getFullYear();

  const navigationLinks = [
    { path: '/interactive-landing-hub', label: 'Home' },
    { path: '/development-performance', label: 'Model Performance' },
    { path: '/clinical-implementation', label: 'Gene Narrative' },
    { path: '/riskprofile', label: 'Patient Risk Profile Inference' },
    { path: '/analytics', label: 'Analytics' },
  ];

  const socialLinks = [
    { name: 'Github', icon: 'Github', url: 'https://github.com/nikkipikkii' },
    { name: 'Linkedin', icon: 'Linkedin', url: 'https://linkedin.com' },
    { name: 'Mail', icon: 'Mail', url: 'mailto:nandbahukhandi8@gmail.com' },
  ];

  return (
    <footer className="footer-container border-t border-border/40 bg-background">
      <div className="footer-content py-12 md:py-16">
        <div className="footer-grid grid grid-cols-1 md:grid-cols-3 gap-12 lg:gap-16">
          
          {/* Brand & Description */}
          <div className="footer-brand space-y-4">
            <Link to="/interactive-landing-hub" className="footer-logo-link flex items-center gap-2">
              <div className="footer-logo-icon">
                <Icon name="Dna" size={24} color="var(--color-accent)" />
              </div>
              <span className="footer-logo-text font-bold text-xl tracking-tight">OncoRisk</span>
            </Link>
            <p className="footer-description text-sm text-muted-foreground leading-relaxed">
              An individual biotechnology project exploring the intersection of genomic signatures 
              and survival analysis. Developed to bridge the gap between complex genetic data 
              and predictive modeling.
            </p>
            <div className="footer-social flex items-center gap-4">
              {socialLinks?.map((social) => (
                <a
                  key={social?.name}
                  href={social?.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-accent transition-colors"
                  aria-label={social?.name}
                >
                  <Icon name={social?.icon} size={20} />
                </a>
              ))}
            </div>
          </div>

          {/* Researcher Section */}
          <div className="footer-section">
            <h3 className="footer-heading text-sm font-bold uppercase tracking-widest text-foreground mb-6">Researcher</h3>
            <div className="footer-about-me p-4 rounded-xl bg-muted/30 border border-border/50">
              <p className="text-sm font-semibold text-foreground mb-1">Nandini Bahukhandi</p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Biotechnology Student <br />
                Aspiring Bioinformatics Researcher
              </p>
            </div>
            <div className="mt-6 flex items-center gap-2 text-[10px] text-accent font-medium uppercase tracking-tighter">
                <Icon name="FlaskConical" size={14} />
                <span>Academic Research Project</span>
            </div>
          </div>

          {/* Navigation Section */}
          <div className="footer-section">
            <h3 className="footer-heading text-sm font-bold uppercase tracking-widest text-foreground mb-6">Sitemap</h3>
            <nav className="footer-links flex flex-col gap-3">
              {navigationLinks?.map((link) => (
                <Link 
                  key={link?.path} 
                  to={link?.path} 
                  className="text-sm text-muted-foreground hover:text-primary transition-colors w-fit"
                >
                  {link?.label}
                </Link>
              ))}
            </nav>
          </div>

        </div>
      </div>

      {/* Bottom Bar & Disclaimer */}
      <div className="footer-bottom border-t border-border/20 py-8">
        <div className="footer-bottom-content container mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="space-y-2 text-center md:text-left">
            <p className="footer-copyright text-xs text-muted-foreground">
              © {currentYear} OncoRisk by Nandini.
            </p>
            <p className="text-[11px] text-muted-foreground/60 max-w-md italic">
              <strong>Disclaimer:</strong> This project is created for <strong>educational purposes only</strong>. 
              The predictions and analysis provided are part of a student research project and should not 
              be used for medical diagnosis or clinical decision-making.
            </p>
          </div>
          
          <nav className="footer-bottom-links flex items-center gap-6">
              <Link to="/privacy" className="text-xs text-muted-foreground hover:text-foreground">Privacy</Link>
              <Link to="/terms" className="text-xs text-muted-foreground hover:text-foreground">Terms</Link>
          </nav>
        </div>
      </div>
    </footer>
  );
};

export default Footer;