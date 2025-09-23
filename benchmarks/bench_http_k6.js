import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';

export const options = {
  stages: [
    { duration: '10s', target: 50 },   // Ramp up
    { duration: '30s', target: 200 },  // Stay at 200 VUs
    { duration: '10s', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(99)<50'],  // 99% of requests under 50ms
    errors: ['rate<0.01'],             // Error rate under 1%
  },
};

const endpoints = [
  '/healthz',
  '/api/v1/signals',
  '/api/v1/portfolio',
  '/api/v1/orders',
];

export default function () {
  const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
  const url = `${BASE_URL}${endpoint}`;

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'X-Request-ID': `k6-${__VU}-${__ITER}`,
    },
    timeout: '10s',
  };

  let response;

  if (endpoint === '/api/v1/orders') {
    // POST request for orders
    const payload = JSON.stringify({
      symbol: `SYN${String(Math.floor(Math.random() * 100) + 1).padStart(3, '0')}`,
      side: Math.random() > 0.5 ? 'buy' : 'sell',
      quantity: Math.floor(Math.random() * 1000) + 100,
      order_type: 'market',
    });
    response = http.post(url, payload, params);
  } else {
    // GET requests for other endpoints
    response = http.get(url, params);
  }

  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 100ms': (r) => r.timings.duration < 100,
    'has content': (r) => r.body && r.body.length > 0,
  });

  errorRate.add(!success);

  // Random sleep between 0.1 and 0.5 seconds
  sleep(Math.random() * 0.4 + 0.1);
}

export function handleSummary(data) {
  return {
    'stdout': JSON.stringify(data, null, 2),
    'summary.json': JSON.stringify(data, null, 2),
  };
}