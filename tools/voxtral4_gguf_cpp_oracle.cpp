// B2-C2 independent C++ tokenizer oracle.
//
// Uses the real CrispASR voxtral_tekken_vocab.h (pinned + Git-blob verified
// by tools/voxtral4_gguf_cpp_parity.py), not a reimplementation of its
// rank-based merge. The default runtime pre-split below is reproduced from
// CrispASR src/voxtral4b.cpp, which differs from the full Tekken pretokenizer
// implemented in the reference header.
//
// argv[1] = binary sidecar file, argv[2] = runtime|unicode_header,
// argv[3..] = UTF-8 input encoded as lowercase hex.
// Writes one JSON array of token IDs per input.
#include "voxtral_tekken_vocab.h"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

static uint32_t read_u32(std::ifstream& in) {
    unsigned char b[4] = {};
    if (!in.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("short u32");
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) |
           (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}

static uint16_t read_u16(std::ifstream& in) {
    unsigned char b[2] = {};
    if (!in.read(reinterpret_cast<char*>(b), 2)) throw std::runtime_error("short u16");
    return uint16_t(b[0]) | (uint16_t(b[1]) << 8);
}

static std::string read_bytes(std::ifstream& in, size_t n) {
    if (n > 8'000'000) throw std::runtime_error("sidecar length exceeds bound");
    std::string value(n, '\0');
    if (!in.read(value.data(), n)) throw std::runtime_error("short payload");
    return value;
}

static uint8_t unhex(char c) {
    if (c >= '0' && c <= '9') return uint8_t(c - '0');
    if (c >= 'a' && c <= 'f') return uint8_t(c - 'a' + 10);
    throw std::runtime_error("non-lowercase-hex input");
}

static std::string decode_hex(std::string_view hex) {
    if (hex.size() % 2 || hex.size() > 2'000'000)
        throw std::runtime_error("invalid hex argument size");
    std::string result;
    result.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2)
        result.push_back(char((unhex(hex[i]) << 4) | unhex(hex[i + 1])));
    return result;
}

// This is the byte-class pre-tokenization used in CrispASR voxtral4b.cpp.
// It is NOT the fuller Unicode splitter in voxtral_tekken_vocab.h.
static std::vector<std::string> runtime_pre_tokenize(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = (unsigned char)text[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            size_t j = i;
            while (j < text.size() &&
                (text[j] == ' ' || text[j] == '\t' || text[j] == '\n' || text[j] == '\r')) j++;
            out.push_back(text.substr(i, j - i));
            i = j;
        } else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                   (c >= '0' && c <= '9')) {
            size_t j = i;
            while (j < text.size()) {
                unsigned char d = (unsigned char)text[j];
                if ((d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z') ||
                    (d >= '0' && d <= '9')) j++;
                else break;
            }
            out.push_back(text.substr(i, j - i));
            i = j;
        } else if (c >= 0x80) {
            size_t j = i + 1;
            while (j < text.size() && (((unsigned char)text[j] & 0xC0) == 0x80)) j++;
            while (j < text.size() && ((unsigned char)text[j]) >= 0x80) {
                size_t k = j + 1;
                while (k < text.size() && (((unsigned char)text[k] & 0xC0) == 0x80)) k++;
                j = k;
            }
            out.push_back(text.substr(i, j - i));
            i = j;
        } else {
            out.push_back(text.substr(i, 1));
            i++;
        }
    }
    return out;
}

struct Oracle {
    std::vector<std::string> specials;
    std::unordered_map<std::string, int32_t> bytes_to_rank;
    uint32_t embedding_rows;

    explicit Oracle(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("cannot open sidecar");
        embedding_rows = read_u32(file);
        uint32_t special_count = read_u32(file);
        if (embedding_rows != 131072 || special_count != 1000)
            throw std::runtime_error("unqualified model geometry");
        for (uint32_t i = 0; i < special_count; ++i)
            specials.push_back(read_bytes(file, read_u16(file)));
        uint32_t ranks = read_u32(file);
        uint32_t length = read_u32(file);
        if (ranks != 150000 || length > 4'000'000)
            throw std::runtime_error("unqualified Tekken BPE data");
        const auto raw = read_bytes(file, length);
        if (file.peek() != EOF) throw std::runtime_error("unexpected sidecar tail");
        std::vector<uint8_t> bytes(raw.begin(), raw.end());
        std::vector<uint32_t> offsets, lengths;
        voxtral_tekken::index_blob_ranks(bytes, int(ranks), offsets, lengths);
        if (offsets.size() != ranks)
            throw std::runtime_error("truncated BPE rank index");
        int limit = voxtral_tekken::active_bpe_count(int(embedding_rows), int(special_count));
        if (limit != 130072)
            throw std::runtime_error("unexpected active BPE limit");
        int inactive = voxtral_tekken::build_rank_map(bytes, offsets, lengths, limit, bytes_to_rank);
        if (inactive != 19928) throw std::runtime_error("incorrect inactive count");
    }

    std::vector<int32_t> encode(const std::string& input, bool unicode_header) const {
        std::vector<int32_t> result;
        size_t pos = 0;
        while (pos < input.size()) {
            bool matched = false;
            for (size_t i = 0; i < specials.size(); ++i) {
                const auto& special = specials[i];
                if (!special.empty() && input.compare(pos, special.size(), special) == 0) {
                    result.push_back(int32_t(i));
                    pos += special.size();
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
            size_t next = input.size();
            for (const auto& special : specials) {
                if (special.empty()) continue;
                const size_t index = input.find(special, pos);
                if (index != std::string::npos && index < next) next = index;
            }
            if (next <= pos) throw std::runtime_error("special token scanner stalled");
            const std::string part = input.substr(pos, next - pos);
            const auto pre = unicode_header ? voxtral_tekken::pre_tokenize(part)
                                            : runtime_pre_tokenize(part);
            for (const auto& token : pre) {
                voxtral_tekken::bpe_encode_ranked(
                    bytes_to_rank, int(specials.size()),
                    reinterpret_cast<const uint8_t*>(token.data()), token.size(), result
                );
            }
            pos = next;
        }
        for (auto id : result)
            if (id < 0 || uint32_t(id) >= embedding_rows)
                throw std::runtime_error("reference emitted an out-of-range token ID");
        return result;
    }
};

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: oracle sidecar runtime|unicode_header hex...");
        Oracle oracle(argv[1]);
        std::string mode(argv[2]);
        if (mode != "runtime" && mode != "unicode_header")
            throw std::runtime_error("unrecognized reference pre-tokenizer mode");
        for (int index = 3; index < argc; ++index) {
            auto ids = oracle.encode(decode_hex(argv[index]), mode == "unicode_header");
            std::cout << "[";
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << ids[i];
            }
            std::cout << "]\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "B2C2_ORACLE_ERROR " << error.what() << "\n";
        return 2;
    }
    return 0;
}
