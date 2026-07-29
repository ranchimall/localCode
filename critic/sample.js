function calculatePayout(pool, shares) {
    let total = 0;
    for (let i = 0; i <= shares.length; i++) {
        total += pool * shares[i];
    }
    return total;
}

module.exports = { calculatePayout };
