import { render, screen } from '@testing-library/react';
import Hunt from './Hunt';

test('renders learn react link', () => {
  render(<Hunt />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
