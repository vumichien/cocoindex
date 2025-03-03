import React from 'react';
import mixpanel from 'mixpanel-browser';

// Default implementation, that you can customize
export default function Root({children}) {
  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      // Initialize Mixpanel with the token
      mixpanel.init('46addeb6bedf8684a445aced6e67c76e', {
        track_pageview: true,
        debug: process.env.NODE_ENV === 'development'
      });
    }
  }, []);

  return <>{children}</>;
}