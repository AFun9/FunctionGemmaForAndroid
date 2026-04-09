package com.gemma.functiongemma.android;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

public class PrefixCacheIndexStoreTest {
    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void upsertAndFindRoundTripsEntry() throws Exception {
        PrefixCacheIndexStore store = new PrefixCacheIndexStore(temporaryFolder.newFolder("cache"));

        PrefixCacheEntry entry = new PrefixCacheEntry();
        entry.toolsetId = "music";
        entry.displayName = "Music";
        entry.fingerprint = "abc123";
        entry.kvFileName = "abc123.kvbin";
        entry.systemSequenceLength = 42;
        entry.createdAtEpochMs = 7L;
        store.upsert(entry);

        PrefixCacheEntry loaded = store.find("music", "abc123");

        assertNotNull(loaded);
        assertEquals("Music", loaded.displayName);
        assertEquals("abc123.kvbin", loaded.kvFileName);
        assertEquals(42, loaded.systemSequenceLength);
    }

    @Test
    public void removeDeletesStoredEntry() throws Exception {
        PrefixCacheIndexStore store = new PrefixCacheIndexStore(temporaryFolder.newFolder("cache-remove"));

        PrefixCacheEntry entry = new PrefixCacheEntry();
        entry.toolsetId = "weather";
        entry.displayName = "Weather";
        entry.fingerprint = "fingerprint";
        entry.kvFileName = "fingerprint.kvbin";
        entry.systemSequenceLength = 9;
        entry.createdAtEpochMs = 1L;
        store.upsert(entry);

        PrefixCacheEntry removed = store.remove("weather");

        assertNotNull(removed);
        assertNull(store.find("weather", "fingerprint"));
    }
}
