import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';
import Input from '../../../components/ui/Input';
import { Checkbox } from '../../../components/ui/Checkbox';

const NewsletterSection = () => {
  const [email, setEmail] = useState('');
  const [preferences, setPreferences] = useState({
    newPublications: true,
    projectUpdates: true,
    collaborationOpportunities: false,
    monthlyDigest: false
  });
  const [subscribed, setSubscribed] = useState(false);
  const [error, setError] = useState('');

  const handleSubscribe = (e) => {
    e?.preventDefault();
    setError('');

    if (!email?.trim()) {
      setError('Email address is required');
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/?.test(email)) {
      setError('Please enter a valid email address');
      return;
    }

    const selectedPreferences = Object.entries(preferences)?.filter(([_, value]) => value);
    if (selectedPreferences?.length === 0) {
      setError('Please select at least one notification preference');
      return;
    }

    setSubscribed(true);
    setTimeout(() => {
      setSubscribed(false);
      setEmail('');
      setPreferences({
        newPublications: true,
        projectUpdates: true,
        collaborationOpportunities: false,
        monthlyDigest: false
      });
    }, 3000);
  };

  return (
    <section className="bg-gradient-to-br from-accent/10 via-primary/5 to-transparent border border-accent/20 rounded-lg p-6 md:p-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-12 h-12 md:w-16 md:h-16 bg-accent/20 rounded-full mb-4">
            <Icon name="Mail" size={24} className="text-accent" />
          </div>
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-2">
            Stay Updated with Latest News
          </h2>
          {/* <p className="text-sm md:text-base text-muted-foreground">
            Subscribe to receive notifications about upcoming projects & collaboration Opportunities
          </p> */}
        </div>

        {subscribed ? (
          <div className="bg-success/10 border border-success/20 rounded-lg p-6 text-center animate-fadeIn">
            <Icon name="CheckCircle2" size={48} className="text-success mx-auto mb-3" />
            <h3 className="text-lg font-semibold text-foreground mb-2">Successfully Subscribed!</h3>
            <p className="text-sm text-muted-foreground">
              Thank you for subscribing. You'll receive updates based on your preferences.
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubscribe} className="space-y-6">
            <Input
              type="email"
              label="Email Address"
              placeholder="researcher@university.edu"
              value={email}
              onChange={(e) => setEmail(e?.target?.value)}
              error={error && !email?.trim() ? error : ''}
              required
            />

            <div className="space-y-3">
              <label className="block text-sm font-medium text-foreground">
                Notification Preferences
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <Checkbox
                  label="New Publications"
                  description="Get notified when new research is published"
                  checked={preferences?.newPublications}
                  onChange={(e) => setPreferences(prev => ({ ...prev, newPublications: e?.target?.checked }))}
                />
                <Checkbox
                  label="Project Updates"
                  description="Updates on ongoing projects"
                  checked={preferences?.projectUpdates}
                  onChange={(e) => setPreferences(prev => ({ ...prev, projectUpdates: e?.target?.checked }))}
                />
                <Checkbox
                  label="Collaboration Opportunities"
                  description="Invitations to join research projects"
                  checked={preferences?.collaborationOpportunities}
                  onChange={(e) => setPreferences(prev => ({ ...prev, collaborationOpportunities: e?.target?.checked }))}
                />
                <Checkbox
                  label="Monthly Digest"
                  description="Comprehensive monthly research summary"
                  checked={preferences?.monthlyDigest}
                  onChange={(e) => setPreferences(prev => ({ ...prev, monthlyDigest: e?.target?.checked }))}
                />
              </div>
              {error && Object.values(preferences)?.every(v => !v) && (
                <p className="text-sm text-destructive">{error}</p>
              )}
            </div>

            <div className="flex flex-col sm:flex-row gap-3">
              <Button
                type="submit"
                variant="default"
                fullWidth
                iconName="Mail"
                iconPosition="left"
              >
                Subscribe to Updates
              </Button>
            </div>

            <p className="text-xs text-muted-foreground text-center">
              By subscribing, you agree to receive research updates. You can unsubscribe at any time. 
              We respect your privacy and will never share your information.
            </p>
          </form>
        )}
      </div>
    </section>
  );
};

export default NewsletterSection;