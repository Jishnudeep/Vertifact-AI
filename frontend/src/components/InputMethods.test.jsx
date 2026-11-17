import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, renderHook, act } from '@testing-library/react';
import { detectInputType, imageProblem } from '../lib/inputType';
import * as api from '../lib/api';
import { useVerify } from '../hooks/useVerify';
import ImagePicker from './ImagePicker';
import ClaimInput from './ClaimInput';
import InputProvenance from './InputProvenance';

const file = (name, type, size) => {
  const f = new File(['x'], name, { type });
  Object.defineProperty(f, 'size', { value: size });
  return f;
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('detectInputType', () => {
  it('a bare link is a url; text containing a link, or a bare scheme, is text', () => {
    expect(detectInputType('  https://www.bbc.com/news/articles/abc  ')).toBe('url');
    expect(detectInputType('Read https://bbc.com/x it says the moon is cheese')).toBe('text');
    expect(detectInputType('http://')).toBe('text');
    expect(detectInputType('http://localhost')).toBe('text');
    expect(detectInputType('NASA found water on the Moon')).toBe('text');
  });

  it('imageProblem accepts PNG/JPEG under 5 MB only', () => {
    expect(imageProblem(file('a.png', 'image/png', 1000))).toBeNull();
    expect(imageProblem(file('a.gif', 'image/gif', 1000))).toMatch(/PNG or JPEG under 5 MB/);
    expect(imageProblem(file('a.jpg', 'image/jpeg', 6 * 1024 * 1024))).toMatch(/under 5 MB/);
  });
});

describe('ImagePicker', () => {
  it('rejects a gif and a 6 MB file before any request', () => {
    const onPick = vi.fn();
    const onReject = vi.fn();
    render(<ImagePicker onPick={onPick} onReject={onReject} />);
    const input = screen.getByTestId('image-input');
    fireEvent.change(input, { target: { files: [file('a.gif', 'image/gif', 10)] } });
    fireEvent.change(input, { target: { files: [file('b.png', 'image/png', 6 * 1024 * 1024)] } });
    expect(onReject).toHaveBeenCalledTimes(2);
    expect(onPick).not.toHaveBeenCalled();
  });

  it('a rejected file shows the copy in ClaimInput and nothing is submitted', () => {
    const onSubmitImage = vi.fn();
    render(<ClaimInput value="" onSubmitImage={onSubmitImage} />);
    fireEvent.change(screen.getByTestId('image-input'), { target: { files: [file('a.gif', 'image/gif', 10)] } });
    expect(screen.getByText('Upload a PNG or JPEG under 5 MB.')).toBeInTheDocument();
    expect(onSubmitImage).not.toHaveBeenCalled();
  });

  it('a blocked URL error adds the paste-the-headline fallback', () => {
    const error = { status: 400, code: 'invalid_input', detail: 'Could not read that page.' };
    render(<ClaimInput value="https://x.com/a" error={error} inputKind="url" />);
    expect(screen.getByText('Could not read that page. Paste the headline or the claim instead.')).toBeInTheDocument();
  });
});

describe('API and hook', () => {
  it('verifyImage posts a multipart "image" field without a JSON content type', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ claim: 'x' }) });
    vi.stubGlobal('fetch', fetchMock);
    const png = file('s.png', 'image/png', 10);
    await api.verifyImage(png);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/api\/verify\/image$/);
    expect(init.body).toBeInstanceOf(FormData);
    expect(init.body.get('image')).toBe(png);
    expect(init.headers['Content-Type']).toBeUndefined();
  });

  it('submit sends a bare link as input_type url and exposes the kind', async () => {
    const spy = vi.spyOn(api, 'verifyClaim').mockResolvedValue({ id: null, claim: 'h' });
    const { result } = renderHook(() => useVerify());
    await act(async () => {
      await result.current.submit(' https://www.bbc.com/news/x ');
    });
    expect(spy).toHaveBeenCalledWith('https://www.bbc.com/news/x', 'url');
    expect(result.current.inputKind).toBe('url');
  });

  it('submitImage calls verifyImage and sets the image kind', async () => {
    const spy = vi.spyOn(api, 'verifyImage').mockResolvedValue({ id: null, claim: 'h' });
    const { result } = renderHook(() => useVerify());
    const png = file('s.png', 'image/png', 10);
    await act(async () => {
      await result.current.submitImage(png);
    });
    expect(spy).toHaveBeenCalledWith(png);
    expect(result.current.inputKind).toBe('image');
  });
});

describe('review fixes', () => {
  it('retry replays the last screenshot, not the text box', async () => {
    const png = file('s.png', 'image/png', 10);
    const spy = vi
      .spyOn(api, 'verifyImage')
      .mockRejectedValueOnce(Object.assign(new Error('down'), { status: 503 }))
      .mockResolvedValueOnce({ id: null, claim: 'h' });
    const textSpy = vi.spyOn(api, 'verifyClaim');
    const { result } = renderHook(() => useVerify());
    await act(async () => {
      await result.current.submitImage(png);
    });
    expect(result.current.status).toBe('error');
    await act(async () => {
      await result.current.retry();
    });
    expect(spy).toHaveBeenCalledTimes(2);
    expect(spy).toHaveBeenLastCalledWith(png);
    expect(textSpy).not.toHaveBeenCalled();
    expect(result.current.status).toBe('done');
  });

  it('dropping a valid image submits it; an invalid one shows the copy instead', () => {
    const onSubmitImage = vi.fn();
    render(<ClaimInput value="" onSubmitImage={onSubmitImage} />);
    const zone = screen.getByLabelText('Claim or article link').parentElement;
    const drop = (f) => fireEvent.drop(zone, { dataTransfer: { types: ['Files'], files: [f] } });
    drop(file('a.gif', 'image/gif', 10));
    expect(screen.getByRole('alert')).toHaveTextContent('Upload a PNG or JPEG under 5 MB.');
    const png = file('s.png', 'image/png', 10);
    drop(png);
    expect(onSubmitImage).toHaveBeenCalledWith(png);
  });

  it('a drop while checking is ignored', () => {
    const onSubmitImage = vi.fn();
    render(<ClaimInput value="" status="checking" onSubmitImage={onSubmitImage} />);
    const zone = screen.getByLabelText('Claim or article link').parentElement;
    fireEvent.drop(zone, { dataTransfer: { types: ['Files'], files: [file('s.png', 'image/png', 10)] } });
    expect(onSubmitImage).not.toHaveBeenCalled();
  });
});

describe('InputProvenance', () => {
  it('never links a non-http source_url', () => {
    const { container } = render(
      <InputProvenance result={{ claim: 'h', source_url: 'javascript://example.com/%0aalert(1)' }} />
    );
    expect(container.querySelector('a')).toBeNull();
  });

  it('says when the page was only partly readable', () => {
    render(<InputProvenance result={{ claim: 'h', source_url: 'https://x.com/a', scrape_degraded: true }} />);
    expect(screen.getByText(/headline only/)).toBeInTheDocument();
  });

  it('url: links the source domain in a new tab with noopener', () => {
    render(<InputProvenance result={{ claim: 'h', source_url: 'https://www.theguardian.com/a/b' }} />);
    const link = screen.getByRole('link', { name: 'theguardian.com' });
    expect(link).toHaveAttribute('href', 'https://www.theguardian.com/a/b');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('image: shows confidence and an edit action with the read text', () => {
    const onEditClaim = vi.fn();
    render(
      <InputProvenance
        result={{ claim: 'Coffee cures cancer', ocr_extracted_text: 'Coffee cures cancer', ocr_confidence: 0.869 }}
        onEditClaim={onEditClaim}
      />
    );
    expect(screen.getByText(/87% confidence/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Edit and re-check' }));
    expect(onEditClaim).toHaveBeenCalledWith('Coffee cures cancer');
  });

  it('text, or a stored result without provenance, renders nothing', () => {
    const { container } = render(<InputProvenance result={{ claim: 'plain' }} />);
    expect(container).toBeEmptyDOMElement();
  });
});
